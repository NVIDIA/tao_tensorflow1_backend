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
"""Patch DynamicGraph."""
from collections import OrderedDict
import copy
from graphsurgeon import StaticGraph
from graphsurgeon._utils import _get_node_names, _handle_single_nodes
from tensorflow import NodeDef

try:
    basestring  # noqa pylint: disable=E0601
except NameError:
    basestring = str


class DynamicGraph(StaticGraph):
    '''
    A sub-class of StaticGraph that can search and modify a TensorFlow GraphDef.

    Args:
        graphdef (tensorflow.GraphDef/tensorflow.Graph OR graphsurgeon.StaticGraph/
                  graphsurgeon.DynamicGraph OR str): A TensorFlow GraphDef/Graph or
        a StaticGraph/DynamicGraph from which to construct this graph, or a string
        containing the path to a frozen model.
    '''

    '''Graph Analysis Functions'''
    # Finds nodes in the graph that would be unused if a certain set of nodes were removed.
    # The returned list includes the nodes provided to the function.
    def _find_unused_nodes_on_removal(self, node_removal_list):
        # Since node_outputs will be modified, need a local copy
        node_outputs = copy.deepcopy(self.node_outputs)

        def recursively_remove_inputs(node):
            # Given one node, return a set containing it and all its hanging inputs
            removable_nodes_list = [node]
            for input_name in node.input:
                # Remove this node from the output of it's inputs
                if input_name in node_outputs and node in node_outputs[input_name]:
                    node_outputs[input_name].remove(node)
                # Recursively remove any inputs which are left hanging
                if input_name not in node_outputs or len(node_outputs[input_name]) == 0:
                    input_name = input_name.replace('^', '').split(':')[0]
                    input_node = self.node_map[input_name]
                    removable_nodes_list.extend(recursively_remove_inputs(input_node))
            return removable_nodes_list

        # Nodes that can be removed based on nodes going to be removed.
        removable_nodes_list = []
        for node in node_removal_list:
            removable_nodes_list.extend(recursively_remove_inputs(node))
        return removable_nodes_list

    '''Graph Manipulation Functions'''
    # Given a graphdef and a container of node names, generates a new graph with all the
    # inputs of the specified nodes recursively forwarded, and the nodes themselves removed.
    def _forward_inputs_impl(self, forward_inputs_names):
        nodes = self._internal_graphdef.node

        # FIXME: Handle control inputs properly when bridging.
        # Figure out the duplicate input situation.
        def should_forward_inputs(node):
            # Forward inputs if the node is in the list...
            is_in_forward_inputs_names = node.name in forward_inputs_names
            # ...unless it has control edge inputs
            has_control_edge = False
            for input_name in node.input:
                if '^' in input_name:
                    has_control_edge = True
            return is_in_forward_inputs_names and not has_control_edge

        def generate_input_replacements():
            def generate_shallow_input_replacements():
                shallow_input_replacements = OrderedDict()
                # Traverse the graph once to get a shallow mapping of input -> replacements
                for node in nodes:
                    if should_forward_inputs(node):
                        shallow_input_replacements[node.name] = node.input
                return shallow_input_replacements

            # Initial pass to get 1-layer deep replacements.
            shallow_input_replacements = generate_shallow_input_replacements()
            # Traverse the input replacement map and generate a map of true input replacements.
            for node_name in shallow_input_replacements:
                for input_name in shallow_input_replacements[node_name]:
                    if input_name in shallow_input_replacements:
                        # Append replacements to the end of the input list
                        shallow_input_replacements[node_name].extend(
                            shallow_input_replacements[input_name])
                        # Pop replaced inputs from the front.
                        shallow_input_replacements[node_name].remove(input_name)
            # Done!
            return shallow_input_replacements

        def update_inputs(node, true_input_replacements):
            # Update inputs, replacing those which need to be.
            def get_replaced_input(input_name):
                # REVIEW: Might need to do this a different way later.
                # Check the true input name, not just as a control input.
                new_input_name = input_name.replace('^', '')
                if new_input_name in true_input_replacements:
                    return new_input_name
                return None

            index = 0
            while index < len(node.input):
                # REVIEW: Might need to do this a different way later.
                input_name = get_replaced_input(node.input[index])
                if input_name:
                    # REVIEW: Do we need to check for unique inputs here?
                    # unique_replacement_names = [replacement_name
                    #     for replacement_name in true_input_replacements[input_name]
                    #         if replacement_name not in new_node.input]

                    # Remove the old input, replace with the new ones.
                    # Make sure to insert in the correct spot,
                    # so as to preserve input ordering.
                    for replacement in true_input_replacements[input_name]:
                        node.input.insert(index, replacement)
                        index += 1
                    del node.input[index]
                    index -= 1
                index += 1

        # Get true replacements.
        true_input_replacements = generate_input_replacements()
        # Update the graph.
        index = 0
        while index < len(nodes):
            if should_forward_inputs(nodes[index]):
                # If this node should be forwarded, remove it.
                del nodes[index]
                index -= 1
            else:
                # For all other nodes, update their inputs with replacements.
                update_inputs(nodes[index], true_input_replacements)
            index += 1

    # Given a graph def, removes nodes corresponding to the names provided and
    # returns a new GraphDef. Does not forward inputs.
    def _remove_impl(self, remove_names):
        nodes = self._internal_graphdef.node

        def should_remove_node_name(node_name):
            # Determine whether this node_name should be removed from the graph
            node_name = node_name.replace('^', '')
            should_remove_node = node_name in remove_names
            # Check if this node shows up as a control dependency
            should_remove_control_dependency = '^' + node_name in remove_names
            return should_remove_node or should_remove_control_dependency

        def update_inputs(node):
            # Update inputs in the node, removing where necessary.
            index = 0
            while index < len(node.input):
                if should_remove_node_name(node.input[index]):
                    del node.input[index]
                    index -= 1
                index += 1

        # Update the graph.
        index = 0
        while index < len(nodes):
            if should_remove_node_name(nodes[index].name):
                del nodes[index]
                index -= 1
            else:
                # Remove the deleted nodes from the inputs of other nodes.
                update_inputs(nodes[index])
            index += 1

    # Given tensorflow GraphDef and a dict of namespace names -> plugin names,
    # collapses those namespaces into single nodes representing plugins, excluding
    # those nodes specified in exclude_nodes.
    def _collapse_namespaces_impl(self, namespace_map, exclude_node_names, unique_inputs):
        nodes = self._internal_graphdef.node
        # TODO: Maybe let this function arbitrarily collapse any group of nodes.
        # Will require more work on user end to collapse multiple namespaces if
        # implemented this way, but provides much greater flexibility. Maybe some
        # compromise is possible.

        def get_plugin_node(node_name):
            # Get the default plugin node provided by the user, or return None if this
            # does not belong in a plugin.
            if node_name in exclude_node_names:
                # Don't put this node into a plugin, treat as normal node instead.
                return None, ""
            # Check if this node should be omitted from the main graph
            # and return the plugin node if so.
            best_match_depth = -1
            best_match = None
            best_namespace = None
            for namespace in namespace_map:
                # Find the end point of the namespace
                current_depth = len(namespace.split('/'))
                # Get a section of the node path to the same depth
                node_namespace = "/".join(node_name.split('/')[:current_depth])
                # Try to match to the longest possible namespace path,
                # then make sure it actually is a path.
                if namespace == node_namespace and current_depth > best_match_depth:
                    best_match_depth = current_depth
                    best_match = namespace_map[namespace]
                    best_namespace = namespace
            return best_match, best_namespace

        def update_inputs(node):
            index = 0
            while index < len(node.input):
                input_name = node.input[index].replace('^', '')
                # We don't care if this is a control input for the purposes of plugins.
                # (That's what the ^ indicates).
                input_plugin, _ = get_plugin_node(input_name)
                # If this input is in a plugin, replace with the plugin name instead.
                if input_plugin:
                    # Remove and replace the node
                    del node.input[index]
                    if input_plugin.name not in node.input:
                        # For plugin inputs, don't add duplicates.
                        node.input.insert(index, input_plugin.name)
                    else:
                        index -= 1
                index += 1

        def update_plugin_inputs(plugin_node, node):
            def add_input(plugin_node, input_name):
                if not unique_inputs or input_name not in plugin_node.input:
                    # If we're not checking for unique inputs, we can add the input all the time.
                    # Otherwise, the input must not already be present.
                    plugin_node.input.append(input_name)

            for input_name in node.input:
                # We don't care if this is a control input for the purposes of plugins.
                # (That's what the ^ indicates).
                input_plugin, _ = get_plugin_node(input_name.replace('^', ''))
                # If the input is in a plugin, we need to add the plugin instead.
                if input_plugin:
                    # If it's in the same plugin, it's not really an input;
                    # otherwise, we can add it.
                    if input_plugin.name != plugin_node.name:
                        add_input(plugin_node, input_plugin.name)
                else:
                    # And if it's not in a plugin, just add it as a normal node.
                    add_input(plugin_node, input_name)

        # Update the graph.
        index = 0
        while index < len(nodes):
            plugin_node, plugin_namespace = get_plugin_node(nodes[index].name)
            if plugin_node:
                # Add the inputs of this node to its plugin.
                update_plugin_inputs(namespace_map[plugin_namespace], nodes[index])
                # Finally, remove it from the main graph.
                del nodes[index]
                index -= 1
            else:
                # For non-plugin nodes, just update their inputs.
                update_inputs(nodes[index])
            index += 1
        # Then integrate the plugin nodes back into the graph.
        temp_l = []
        nm = set()
        for node in namespace_map.values():
            if (node.name not in nm):
                temp_l.append(node)
                nm.add(node.name)
        nodes.extend(temp_l)

    def _iterable(self, buf):
        '''
        Checks whether buf is a iterable instance.

        Returns:
           buf if `buf` is iterable
           [buf] otherwise
        '''
        iterable = None
        try:
            iter(buf)
            iterable = True
        except TypeError:
            iterable = False
        if (isinstance(buf, basestring)):
            iterable = False
        if (isinstance(buf, NodeDef)):
            iterable = False
        if not iterable:
            buf = [buf]
        return buf

    def _force_to_names(self, buf):
        '''
        Converts a given list or singleton to represents name(s) of vertices in the graph.

        Args:
           buf - a list of nodes or node names or
               a singleton node or node name

        Returns:
           A list of names corresponding to `buf`. This always
           returns a list even if the argument was a singleton
        '''

        buf2 = []
        nm = set()
        for node in self:
            nm.add(node.name)
        buf = self._iterable(buf)
        for x in buf:
            el = None
            if (isinstance(x, basestring)):
                el = x
            elif (isinstance(x, NodeDef)):
                el = x.name
            else:
                assert False, "The iterable list `buf` must \
                               consists of either names or tf.NodeDefs"
            if el in nm:
                buf2.append(el)
            else:
                assert False, "The name %s does not exist" % el
        return buf2

    def _force_to_nodes(self, buf):
        '''
        Converts a given list or singleton to represent node object(s) of vertices in the graph.

        Args:
           buf - a list of nodes or node names or
               a singleton node or node name

        Returns:
           A list of node objects corresponding to `buf`. This always
           returns a list even if the argument was a singleton
        '''

        buf = self._force_to_names(buf)
        buf2 = []
        nm = {}
        for node in self:
            nm[node.name] = node
        buf = self._iterable(buf)
        for x in buf:
            el = None
            if (isinstance(x, basestring)):
                if (x not in nm):
                    assert False, "The name %s does not exist" % x
                else:
                    el = nm[x]
            elif (isinstance(x, NodeDef)):
                el = x
            else:
                assert False, "The iterable list `buf` must \
                               consists of either names or tf.NodeDefs"
            buf2.append(el)
        return buf2

    def sort(self, name, inputs, error_if_not_found=True):
        '''
        Mpves particular nodes to the front of the input list of node `name`.

        The ingoing edges to `name` that are in `input` are moved to the front
        (they are placed in the same order as `input`) and the rest
        of edges are placed on the back (in the preserved order)

        Args:
            name - node or node name to the node which input will change
            input - a desired move of edges

        Returns:
            None
        '''

        node, = self._force_to_nodes(name)
        inputs = self._force_to_names(inputs)

        for edge in node.input:
            if edge not in inputs:
                inputs.append(edge)
        for edge in inputs:
            if (edge not in node.input):
                if (error_if_not_found):
                    assert False, "Node %s was not found" % edge
            else:
                node.input.remove(edge)
        for edge in inputs:
            node.input.append(edge)

    def add(self, nodes):
        '''
        Adds free-standing nodes to the graph.

        Args:
            nodes - list of nodes or node names or
                    a singleton node or node name

        Returns:
            None
        '''
        nodes = self._iterable(nodes)
        for node in nodes:
            assert isinstance(node, NodeDef), "Nodes that are being \
                                               added to the graph must be \
                                               of instance tf.NodeDef"
            for elem in self:
                if (elem.name == node.name):
                    assert False, "Node %s already in the graph" % node.name
            self._internal_graphdef.node.extend([node])

    def connect(self, who, to_whom, error_if_connected=False):
        '''
        Connects two nodes.

        `who` is connected to `to_whom` so the node
        `to_whom` will have an ingoing edge from `who`

        Args:
            who, to_whom - nodes or node names

        Returns:
            None
        '''
        who, to_whom, = self._force_to_nodes([who, to_whom])
        print(who.name, to_whom.name)
        who_name, = self._force_to_names(who)
        if (who_name in to_whom.input) and error_if_connected:
            assert False, "Vertices %s (connecting `who`) and %s \
                          (connecting `to_whom`) are already connected" \
                        % (who, to_whom)
        elif not (who_name in to_whom.input):
            to_whom.input.append(who_name)

    def disconnect(self, who, of_whom, error_if_not_connected=False):
        '''
        Disconnects two nodes.

        `who` is disconnected from `of_whom` so the
        ingoing edge in node `of_whom` is removed

        Args:
            who, of_whom - nodes in the graph

        Returns:
            None
        '''
        who, of_whom, = self._force_to_nodes([who, of_whom])
        who_name, = self._force_to_names(who)
        if (not (who_name in of_whom.input)) and error_if_not_connected:
            assert False, "Vertices %s (disconnecting `who`) \
                           and %s (disconnecting `of_whom`) \
                           are not connected" % (who, of_whom)
        elif (who_name in of_whom.input):
            of_whom.input.remove(who_name)

    # Wrapper to handle exclude_nodes
    def collapse_namespaces(self, namespace_map, exclude_nodes=None, unique_inputs=True):
        '''
        Collapses nodes in namespaces.

        Args:
            namespace_map (dict(str, tensorflow.NodeDef)): A dictionary specifying namespaces
                and their corresponding plugin nodes. These plugin nodes are typically used to
                 specify attributes of the custom plugin, while inputs and outputs
                 are automatically deduced.
                 Multiple namespaces can be collapsed into a single plugin node, and
                 nested namespaces are collapsed into plugin nodes outside their
                 parent namespaces.
            exclude_nodes (list(tensorflow.NodeDef)): Iterable container (usually a list) of nodes
                which should NOT be collapsed. These nodes will be present in the final graph as
                either inputs or outputs of the plugin nodes.
            unique_inputs (bool): Whether inputs to the collapsed node should be unique.
                If this is false, plugin nodes may have duplicate inputs.

        Returns:
            None
        '''
        exclude_node_names = set(_get_node_names(exclude_nodes))
        self._collapse_namespaces_impl(namespace_map, exclude_node_names, unique_inputs)
        # After modifying, need to regenerate analysis data.
        # TODO: Remove this, and do it more efficiently during traversal.
        self._initialize_analysis_data()

    # Allows for removal of nodes based on node references directly.
    def remove(self, nodes, remove_exclusive_dependencies=False):
        '''
        Removes nodes from this graph.

        Args:
            nodes: A list of nodes or node names or
                   a singleton node or node name to be removed
            remove_exclusive_dependencies (bool): Whether to also remove dependencies exclusive
                to the nodes about to be removed. When set to True, all exclusive dependencies
                will be removed recursively, and the number of hanging nodes in the graph
                will remain constant. Defaults to False.

        Returns:
            None
        '''
        nodes = self._force_to_nodes(nodes)
        nodes = _handle_single_nodes(nodes)
        if remove_exclusive_dependencies:
            nodes = self._find_unused_nodes_on_removal(nodes)
        remove_names = set(_get_node_names(nodes))
        # The implementation requires node names, rather than references.
        self._remove_impl(remove_names)
        # After modifying, need to regenerate analysis data.
        # TODO: Remove this, and do it more efficiently during traversal.
        self._initialize_analysis_data()

    # Allows for removal of nodes based on node references directly.
    def forward_inputs(self, nodes):
        '''
        Removes nodes from this graph.

        **Warning**: Nodes with control inputs are not removed, so as not to break the structure of
            the graph. If you need to forward these, remove their control inputs first.

        Args:
            nodes (list(tensorflow.NodeDef))): Iterable container (usually a list) of nodes which
                should be removed and whose inputs forwarded.

        Returns:
            None
        '''
        nodes = _handle_single_nodes(nodes)
        forward_inputs_names = set(_get_node_names(nodes))
        # The implementation requires node names, rather than references.
        self._forward_inputs_impl(forward_inputs_names)
        # After modifying, need to regenerate analysis data.
        # TODO: Remove this, and do it more efficiently during traversal.
        self._initialize_analysis_data()

    def extend(self, node_list):
        '''
        Extends this graph's nodes based on the provided list.

        Args:
            node_list (list(tensorflow.NodeDef)): List of TensorFlow NodeDefs to add to the graph.

        Returns:
            None
        '''
        self._internal_graphdef.node.extend(node_list)

    def append(self, node):
        '''
        Appends a node to this graph.

        Args:
            node (tensorflow.NodeDef): TensorFlow NodeDef to add to the graph.

        Returns:
            None
        '''
        self._internal_graphdef.node.extend([node])
