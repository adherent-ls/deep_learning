import copy
import types

import yaml
from yaml import Token, AnchorToken, TagToken, BlockEntryToken, SequenceStartEvent, ScalarToken, ScalarEvent, \
    FlowSequenceStartToken, FlowMappingStartToken, MappingStartEvent, BlockSequenceStartToken, BlockMappingStartToken, \
    AliasToken, AliasEvent, FlowEntryToken, KeyToken, CollectionStartEvent, CollectionEndEvent, SequenceNode, NodeEvent, \
    Node, ScalarNode, MappingNode
from yaml.composer import Composer, ComposerError
from yaml.constructor import Constructor, ConstructorError
from yaml.parser import Parser, ParserError
from yaml.reader import Reader
from yaml.resolver import Resolver
from yaml.scanner import Scanner, ScannerError


class FunctionToken(Token):
    id = '<function>'

    def __init__(self, value, start_mark, end_mark):
        super(FunctionToken, self).__init__(start_mark, end_mark)
        self.value = value


class FlowParamStartToken(Token):
    id = '('


class FlowParamEndToken(Token):
    id = ')'


class ParamStartEvent(CollectionStartEvent):
    def __init__(self, anchor, tag, function, implicit, start_mark=None, end_mark=None,
                 flow_style=None):
        super(ParamStartEvent, self).__init__(anchor, tag, implicit, start_mark, end_mark, flow_style)
        self.function = function


class ParamEndEvent(CollectionEndEvent):
    pass


class FunctionEvent(NodeEvent):
    def __init__(self, anchor, tag, function, implicit, value,
                 start_mark=None, end_mark=None, style=None):
        self.anchor = anchor
        self.tag = tag if tag is not None else 'tag:yaml.org,2002:function'
        self.function = function
        self.implicit = implicit
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.style = style


class FunctionNode(Node):
    id = 'function'

    def __init__(self, tag, function, value,
                 start_mark=None, end_mark=None, style=None):
        self.tag = tag if tag is not None else 'tag:yaml.org,2002:function'
        self.function = function
        self.value = value
        # self.value = eval(function)(*value) if function != '' else eval(*value)
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.style = style


class YamlScanner(Scanner):
    def __init__(self):
        super(YamlScanner, self).__init__()

    def fetch_more_tokens(self):

        # Eat whitespaces and comments until we reach the next token.
        self.scan_to_next_token()

        # Remove obsolete possible simple keys.
        self.stale_possible_simple_keys()

        # Compare the current indentation and column. It may add some tokens
        # and decrease the current indentation level.
        self.unwind_indent(self.column)

        # Peek the next character.
        ch = self.peek()

        # Is it the end of stream?
        if ch == '\0':
            return self.fetch_stream_end()

        # Is it a directive?
        if ch == '%' and self.check_directive():
            return self.fetch_directive()

        # Is it the document start?
        if ch == '-' and self.check_document_start():
            return self.fetch_document_start()

        # Is it the document end?
        if ch == '.' and self.check_document_end():
            return self.fetch_document_end()

        # TODO: support for BOM within a stream.
        # if ch == '\uFEFF':
        #    return self.fetch_bom()    <-- issue BOMToken

        # Note: the order of the following checks is NOT significant.

        # Is it the flow sequence start indicator?
        if ch == '[':
            return self.fetch_flow_sequence_start()

        # Is it the flow mapping start indicator?
        if ch == '{':
            return self.fetch_flow_mapping_start()

        # Is it the flow param start indicator?
        if ch == '(':
            return self.fetch_flow_param_start()

        # Is it the flow sequence end indicator?
        if ch == ']':
            return self.fetch_flow_sequence_end()

        # Is it the flow mapping end indicator?
        if ch == '}':
            return self.fetch_flow_mapping_end()

        # Is it the flow param end indicator?
        if ch == ')':
            return self.fetch_flow_param_end()

        # Is it the flow entry indicator?
        if ch == ',':
            return self.fetch_flow_entry()

        # Is it the block entry indicator?
        if ch == '-' and self.check_block_entry():
            return self.fetch_block_entry()

        # Is it the key indicator?
        if ch == '?' and self.check_key():
            return self.fetch_key()

        # Is it the value indicator?
        if ch == ':' and self.check_value():
            return self.fetch_value()

        # Is it an alias?
        if ch == '*':
            return self.fetch_alias()

        # Is it an anchor?
        if ch == '&':
            return self.fetch_anchor()

        # Is it a function?
        if ch == '$':
            return self.fetch_function()

        # Is it a tag?
        if ch == '!':
            return self.fetch_tag()

        # Is it a literal scalar?
        if ch == '|' and not self.flow_level:
            return self.fetch_literal()

        # Is it a folded scalar?
        if ch == '>' and not self.flow_level:
            return self.fetch_folded()

        # Is it a single quoted scalar?
        if ch == '\'':
            return self.fetch_single()

        # Is it a double quoted scalar?
        if ch == '\"':
            return self.fetch_double()

        # It must be a plain scalar then.
        if self.check_plain():
            return self.fetch_plain()

        # No? It's an error. Let's produce a nice error message.
        raise ScannerError("while scanning for the next token", None,
                           "found character %r that cannot start any token" % ch,
                           self.get_mark())

    def fetch_flow_param_start(self):
        # TODO
        self.fetch_flow_collection_start(FlowParamStartToken)

    def fetch_flow_param_end(self):
        # TODO
        self.fetch_flow_collection_end(FlowParamEndToken)

    def fetch_function(self):
        # TODO
        self.save_possible_simple_key()

        self.allow_simple_key = False
        self.scan_function()

    def fetch_plain(self):

        # A plain scalar could be a simple key.
        self.save_possible_simple_key()

        # No simple keys after plain scalars. But note that `scan_plain` will
        # change this flag if the scan is finished at the beginning of the
        # line.
        self.allow_simple_key = False

        # Scan and add SCALAR. May change `allow_simple_key`.
        self.tokens.append(self.scan_plain())

    def scan_plain(self):
        # See the specification for details.
        # We add an additional restriction for the flow context:
        #   plain scalars in the flow context cannot contain ',' or '?'.
        # We also keep track of the `allow_simple_key` flag here.
        # Indentation rules are loosed for the flow context.
        chunks = []
        start_mark = self.get_mark()
        end_mark = start_mark
        indent = self.indent + 1
        # We allow zero indentation for scalars, but then we need to check for
        # document separators at the beginning of the line.
        # if indent == 0:
        #    indent = 1
        spaces = []
        while True:
            length = 0
            if self.peek() == '#':
                break
            while True:
                ch = self.peek(length)
                if ch in '\0 \t\r\n\x85\u2028\u2029' \
                        or (ch == ':' and
                            self.peek(length + 1) in '\0 \t\r\n\x85\u2028\u2029'
                            + (u',[]{}()' if self.flow_level else u'')) \
                        or (self.flow_level and ch in ',?[]{}()'):
                    break
                length += 1
            if length == 0:
                break
            self.allow_simple_key = False
            chunks.extend(spaces)
            chunks.append(self.prefix(length))
            self.forward(length)
            end_mark = self.get_mark()
            spaces = self.scan_plain_spaces(indent, start_mark)
            if not spaces or self.peek() == '#' \
                    or (not self.flow_level and self.column < indent):
                break
        return ScalarToken(''.join(chunks), True, start_mark, end_mark)

    def scan_anchor(self, TokenClass):
        # The specification does not restrict characters for anchors and
        # aliases. This may lead to problems, for instance, the document:
        #   [ *alias, value ]
        # can be interpreted in two ways, as
        #   [ "value" ]
        # and
        #   [ *alias , "value" ]
        # Therefore we restrict aliases to numbers and ASCII letters.
        start_mark = self.get_mark()
        indicator = self.peek()
        if indicator == '*':
            name = 'alias'
        else:
            name = 'anchor'
        self.forward()
        length = 0
        ch = self.peek(length)
        while '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' \
                or ch in '-_':
            length += 1
            ch = self.peek(length)
        if not length:
            raise ScannerError("while scanning an %s" % name, start_mark,
                               "expected alphabetic or numeric character, but found %r"
                               % ch, self.get_mark())
        value = self.prefix(length)
        self.forward(length)
        ch = self.peek()
        if ch not in '\0 \t\r\n\x85\u2028\u2029?:,]})%@`':
            raise ScannerError("while scanning an %s" % name, start_mark,
                               "expected alphabetic or numeric character, but found %r"
                               % ch, self.get_mark())
        end_mark = self.get_mark()
        return TokenClass(value, start_mark, end_mark)

    def scan_function(self):
        # The specification does not restrict characters for anchors and
        # aliases. This may lead to problems, for instance, the document:
        #   [ *alias, value ]
        # can be interpreted in two ways, as
        #   [ "value" ]
        # and
        #   [ *alias , "value" ]
        # Therefore we restrict aliases to numbers and ASCII letters.
        start_mark = self.get_mark()
        indicator = self.peek()
        if indicator == '*':
            name = 'alias'
        else:
            name = 'anchor'
        self.forward()
        length = 0
        ch = self.peek(length)
        while '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' \
                or ch in '_.':
            length += 1
            ch = self.peek(length)
        if length != 0:
            value = self.prefix(length)
            self.forward(length)
        else:
            value = ''
        # ch = self.peek()
        # if ch not in '\0 \t\r\n\x85\u2028\u2029?:,]}%@`':
        #     raise ScannerError("while scanning an %s" % name, start_mark,
        #                        "expected alphabetic or numeric character, but found %r"
        #                        % ch, self.get_mark())
        end_mark = self.get_mark()
        self.tokens.append(FunctionToken(value, start_mark, end_mark))
        # if value == '':
        #     self.fetch_plain()


class YamlParser(Parser):
    def __init__(self):
        super(YamlParser, self).__init__()

    def parse_node(self, block=False, indentless_sequence=False):
        if self.check_token(AliasToken):
            token = self.get_token()
            event = AliasEvent(token.value, token.start_mark, token.end_mark)
            self.state = self.states.pop()
        else:
            anchor = None
            tag = None
            function = None
            start_mark = end_mark = tag_mark = None
            if self.check_token(AnchorToken):
                token = self.get_token()
                start_mark = token.start_mark
                end_mark = token.end_mark
                anchor = token.value
                if self.check_token(TagToken):
                    token = self.get_token()
                    tag_mark = token.start_mark
                    end_mark = token.end_mark
                    tag = token.value
                if self.check_token(FunctionToken):
                    token = self.get_token()
                    start_mark = token.start_mark
                    end_mark = token.end_mark
                    function = token.value
            elif self.check_token(TagToken):
                token = self.get_token()
                start_mark = tag_mark = token.start_mark
                end_mark = token.end_mark
                tag = token.value
                if self.check_token(AnchorToken):
                    token = self.get_token()
                    end_mark = token.end_mark
                    anchor = token.value
            elif self.check_token(FunctionToken):
                token = self.get_token()
                start_mark = token.start_mark
                end_mark = token.end_mark
                function = token.value
            if tag is not None:
                handle, suffix = tag
                if handle is not None:
                    if handle not in self.tag_handles:
                        raise ParserError("while parsing a node", start_mark,
                                          "found undefined tag handle %r" % handle,
                                          tag_mark)
                    tag = self.tag_handles[handle] + suffix
                else:
                    tag = suffix
            # if tag == '!':
            #    raise ParserError("while parsing a node", start_mark,
            #            "found non-specific tag '!'", tag_mark,
            #            "Please check 'http://pyyaml.org/wiki/YAMLNonSpecificTag' and share your opinion.")
            if start_mark is None:
                start_mark = end_mark = self.peek_token().start_mark
            event = None
            implicit = (tag is None or tag == '!')
            if indentless_sequence and self.check_token(BlockEntryToken):
                end_mark = self.peek_token().end_mark
                event = SequenceStartEvent(anchor, tag, implicit,
                                           start_mark, end_mark)
                self.state = self.parse_indentless_sequence_entry
            else:
                if self.check_token(ScalarToken):
                    token = self.get_token()
                    end_mark = token.end_mark
                    if (token.plain and tag is None) or tag == '!':
                        implicit = (True, False)
                    elif tag is None:
                        implicit = (False, True)
                    else:
                        implicit = (False, False)
                    if function is None:
                        event = ScalarEvent(anchor, tag, implicit, token.value,
                                            start_mark, end_mark, style=token.style)
                    else:
                        event = FunctionEvent(anchor, tag, function, (False, False), token.value,
                                              start_mark, end_mark, style=token.style)
                    self.state = self.states.pop()
                elif self.check_token(FlowSequenceStartToken):
                    end_mark = self.peek_token().end_mark
                    event = SequenceStartEvent(anchor, tag, implicit,
                                               start_mark, end_mark, flow_style=True)
                    self.state = self.parse_flow_sequence_first_entry
                elif self.check_token(FlowMappingStartToken):
                    end_mark = self.peek_token().end_mark
                    event = MappingStartEvent(anchor, tag, implicit,
                                              start_mark, end_mark, flow_style=True)
                    self.state = self.parse_flow_mapping_first_key
                elif block and self.check_token(BlockSequenceStartToken):
                    end_mark = self.peek_token().start_mark
                    event = SequenceStartEvent(anchor, tag, implicit,
                                               start_mark, end_mark, flow_style=False)
                    self.state = self.parse_block_sequence_first_entry
                elif block and self.check_token(BlockMappingStartToken):
                    end_mark = self.peek_token().start_mark
                    event = MappingStartEvent(anchor, tag, implicit,
                                              start_mark, end_mark, flow_style=False)
                    self.state = self.parse_block_mapping_first_key
                elif self.check_token(FlowParamStartToken):
                    end_mark = self.peek_token().end_mark
                    event = ParamStartEvent(anchor, tag, function, implicit,
                                            start_mark, end_mark, flow_style=True)
                    self.state = self.parse_flow_param_first_entry
                elif anchor is not None or tag is not None:
                    # Empty scalars are allowed even if a tag or an anchor is
                    # specified.
                    event = ScalarEvent(anchor, tag, (implicit, False), '',
                                        start_mark, end_mark)
                    self.state = self.states.pop()
                else:
                    if block:
                        node = 'block'
                    else:
                        node = 'flow'
                    token = self.peek_token()
                    raise ParserError("while parsing a %s node" % node, start_mark,
                                      "expected the node content, but found %r" % token.id,
                                      token.start_mark)
        return event

    def parse_flow_param_first_entry(self):
        token = self.get_token()
        self.marks.append(token.start_mark)
        return self.parse_flow_param_entry(first=True)

    def parse_flow_param_entry(self, first=False):
        if not self.check_token(FlowParamEndToken):
            if not first:
                if self.check_token(FlowEntryToken):
                    self.get_token()
                else:
                    token = self.peek_token()
                    raise ParserError("while parsing a flow sequence", self.marks[-1],
                                      "expected ',' or ']', but got %r" % token.id, token.start_mark)

            if self.check_token(KeyToken):
                token = self.peek_token()
                event = MappingStartEvent(None, None, True,
                                          token.start_mark, token.end_mark,
                                          flow_style=True)
                self.state = self.parse_flow_sequence_entry_mapping_key
                return event
            elif not self.check_token(FlowParamEndToken):
                self.states.append(self.parse_flow_param_entry)
                return self.parse_flow_node()
        token = self.get_token()
        event = ParamEndEvent(token.start_mark, token.end_mark)
        self.state = self.states.pop()
        self.marks.pop()
        return event


class YamlComposer(Composer):
    def __init__(self):
        super(YamlComposer, self).__init__()

    def compose_node(self, parent, index):
        if self.check_event(AliasEvent):
            event = self.get_event()
            anchor = event.anchor
            if anchor not in self.anchors:
                raise ComposerError(None, None, "found undefined alias %r"
                                    % anchor, event.start_mark)
            return copy.deepcopy(self.anchors[anchor])
        event = self.peek_event()
        anchor = event.anchor
        if anchor is not None:
            if anchor in self.anchors:
                raise ComposerError("found duplicate anchor %r; first occurrence"
                                    % anchor, self.anchors[anchor].start_mark,
                                    "second occurrence", event.start_mark)
        self.descend_resolver(parent, index)
        if self.check_event(ScalarEvent):
            node = self.compose_scalar_node(anchor)
        elif self.check_event(FunctionEvent):
            node = self.compose_function_node(anchor)
        elif self.check_event(SequenceStartEvent):
            node = self.compose_sequence_node(anchor)
        elif self.check_event(MappingStartEvent):
            node = self.compose_mapping_node(anchor)
        elif self.check_event(ParamStartEvent):
            function = event.function
            node = self.compose_param_node(anchor, function)
        self.ascend_resolver()
        return node

    def compose_function_node(self, anchor):
        event = self.get_event()
        tag = event.tag
        value = [event.value]
        value = eval(*value)
        node = FunctionNode(tag, event.function, value,
                            event.start_mark, event.end_mark, style=event.style)
        if anchor is not None:
            self.anchors[anchor] = node
        return node

    def compose_param_node(self, anchor, function):
        start_event = self.get_event()
        tag = start_event.tag
        values = []
        while not self.check_event(ParamEndEvent):
            values.append(self.construct_document(self.compose_node(None, None)))
        end_event = self.get_event()

        values = self.build_python_instance(function, args=values)
        node = FunctionNode(tag, function, values,
                            start_event.start_mark, end_event.end_mark)
        if anchor is not None:
            self.anchors[anchor] = node
        return node

    def build_python_instance(self, suffix, args=None, kwds=None, newobj=False):
        if not args:
            args = []
        if not kwds:
            kwds = {}
        cls = self.find_python_name(suffix, 0)
        if newobj and isinstance(cls, type):
            return cls.__new__(cls, *args, **kwds)
        else:
            return cls(*args, **kwds)


class YamlConstructor(Constructor):
    def construct_object(self, node, deep=False):
        if node in self.constructed_objects:
            return self.constructed_objects[node]
        if deep:
            old_deep = self.deep_construct
            self.deep_construct = True
        if node in self.recursive_objects:
            raise ConstructorError(None, None,
                                   "found unconstructable recursive node", node.start_mark)
        self.recursive_objects[node] = None
        constructor = None
        tag_suffix = None
        if node.tag in self.yaml_constructors:
            constructor = self.yaml_constructors[node.tag]
        else:
            for tag_prefix in self.yaml_multi_constructors:
                if tag_prefix is not None and node.tag.startswith(tag_prefix):
                    tag_suffix = node.tag[len(tag_prefix):]
                    constructor = self.yaml_multi_constructors[tag_prefix]
                    break
            else:
                if None in self.yaml_multi_constructors:
                    tag_suffix = node.tag
                    constructor = self.yaml_multi_constructors[None]
                elif None in self.yaml_constructors:
                    constructor = self.yaml_constructors[None]
                elif isinstance(node, ScalarNode):
                    constructor = self.__class__.construct_scalar
                elif isinstance(node, SequenceNode):
                    constructor = self.__class__.construct_sequence
                elif isinstance(node, MappingNode):
                    constructor = self.__class__.construct_mapping
                elif isinstance(node, FunctionNode):
                    constructor = self.__class__.construct_function
        if tag_suffix is None:
            data = constructor(self, node)
        else:
            data = constructor(self, tag_suffix, node)
        if isinstance(data, types.GeneratorType):
            generator = data
            data = next(generator)
            if self.deep_construct:
                for dummy in generator:
                    pass
            else:
                self.state_generators.append(generator)
        self.constructed_objects[node] = data
        del self.recursive_objects[node]
        if deep:
            self.deep_construct = old_deep
        return data

    def construct_function(self, node: FunctionNode):
        return node.value


YamlConstructor.add_constructor(
    'tag:yaml.org,2002:function',
    YamlConstructor.construct_function)


class YamlLoader(Reader, YamlScanner, YamlParser, YamlComposer, YamlConstructor, Resolver):
    def __init__(self, stream):
        Reader.__init__(self, stream)
        YamlScanner.__init__(self)
        YamlParser.__init__(self)
        YamlComposer.__init__(self)
        YamlConstructor.__init__(self)
        Resolver.__init__(self)
