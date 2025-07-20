from context_mixer.domain.chunking_engine import (
    generate_chunk_id,
    extract_natural_units,
    classify_unit_type,
    parse_grouping_response,
    create_fallback_groupings,
    fallback_grouping
)


class DescribeGenerateChunkId:
    def should_generate_consistent_id_for_same_content_and_concept(self):
        content = "This is test content"
        concept = "test concept"

        id1 = generate_chunk_id(content, concept)
        id2 = generate_chunk_id(content, concept)

        assert id1 == id2
        assert id1.startswith("chunk_")
        assert len(id1) == 18  # "chunk_" + 12 characters

    def should_generate_different_ids_for_different_content(self):
        concept = "test concept"

        id1 = generate_chunk_id("content 1", concept)
        id2 = generate_chunk_id("content 2", concept)

        assert id1 != id2

    def should_generate_different_ids_for_different_concepts(self):
        content = "same content"

        id1 = generate_chunk_id(content, "concept 1")
        id2 = generate_chunk_id(content, "concept 2")

        assert id1 != id2


class DescribeExtractNaturalUnits:
    def should_extract_single_paragraph(self):
        content = "This is a single paragraph."

        units = extract_natural_units(content)

        assert len(units) == 1
        assert units[0]['content'] == "This is a single paragraph."
        assert units[0]['type'] == 'title'  # Short text is classified as title
        assert units[0]['start_pos'] == 0
        assert units[0]['end_pos'] == 27

    def should_extract_multiple_paragraphs(self):
        content = "First paragraph.\n\nSecond paragraph."

        units = extract_natural_units(content)

        assert len(units) == 2
        assert units[0]['content'] == "First paragraph."
        assert units[1]['content'] == "Second paragraph."

    def should_handle_empty_content(self):
        content = ""

        units = extract_natural_units(content)

        assert len(units) == 0

    def should_skip_empty_paragraphs(self):
        content = "First paragraph.\n\n\n\nSecond paragraph."

        units = extract_natural_units(content)

        assert len(units) == 2
        assert units[0]['content'] == "First paragraph."
        assert units[1]['content'] == "Second paragraph."

    def should_classify_units_correctly(self):
        content = "# Header\n\nRegular paragraph.\n\n```\ncode block\n```"

        units = extract_natural_units(content)

        assert len(units) == 3
        assert units[0]['type'] == 'header'
        assert units[1]['type'] == 'title'  # Short text is classified as title
        assert units[2]['type'] == 'code'


class DescribeClassifyUnitType:
    def should_classify_markdown_headers(self):
        assert classify_unit_type("# Main Header") == 'header'
        assert classify_unit_type("## Sub Header") == 'header'
        assert classify_unit_type("### Sub Sub Header") == 'header'

    def should_classify_code_blocks(self):
        assert classify_unit_type("```python\nprint('hello')\n```") == 'code'
        # Note: indented code gets stripped, so leading spaces are removed
        # This tests the actual behavior after stripping - needs to be >100 chars for paragraph
        long_text = "indented code that is long enough to not be classified as title because it exceeds one hundred characters in length"
        assert classify_unit_type(long_text) == 'paragraph'

    def should_classify_lists(self):
        assert classify_unit_type("- Item 1\n- Item 2") == 'list'
        assert classify_unit_type("* Item 1\n* Item 2") == 'list'
        assert classify_unit_type("1. Item 1\n2. Item 2") == 'list'

    def should_classify_short_text_as_title(self):
        assert classify_unit_type("Short title") == 'title'
        assert classify_unit_type("A" * 50) == 'title'

    def should_classify_long_text_as_paragraph(self):
        long_text = "A" * 150
        assert classify_unit_type(long_text) == 'paragraph'

    def should_classify_multiline_text_as_paragraph(self):
        multiline_text = "Line 1\nLine 2"
        assert classify_unit_type(multiline_text) == 'paragraph'


class DescribeParseGroupingResponse:
    def should_parse_valid_grouping_response(self):
        response = "Here are the groups: [[1, 2], [3, 4, 5], [6]]"
        num_units = 6

        groupings = parse_grouping_response(response, num_units)

        assert groupings == [[0, 1], [2, 3, 4], [5]]

    def should_handle_invalid_response_with_fallback(self):
        response = "Invalid response without proper grouping"
        num_units = 4

        groupings = parse_grouping_response(response, num_units)

        assert len(groupings) > 0
        # Should cover all units
        covered_units = set()
        for group in groupings:
            covered_units.update(group)
        assert covered_units == set(range(num_units))

    def should_handle_partial_grouping_by_adding_missing_units(self):
        response = "Groups: [[1, 2], [4]]"  # Missing unit 3
        num_units = 4

        groupings = parse_grouping_response(response, num_units)

        # Should cover all units
        covered_units = set()
        for group in groupings:
            covered_units.update(group)
        assert covered_units == set(range(num_units))

    def should_filter_out_invalid_unit_indices(self):
        response = "Groups: [[1, 2], [7, 8]]"  # Units 7, 8 don't exist
        num_units = 4

        groupings = parse_grouping_response(response, num_units)

        # Should only include valid indices
        for group in groupings:
            for unit_idx in group:
                assert 0 <= unit_idx < num_units


class DescribeCreateFallbackGroupings:
    def should_group_small_number_of_units_together(self):
        groupings = create_fallback_groupings(3)

        assert groupings == [[0, 1, 2]]

    def should_group_units_in_triples_when_possible(self):
        groupings = create_fallback_groupings(6)

        assert groupings == [[0, 1, 2], [3, 4, 5]]

    def should_handle_remainder_units(self):
        groupings = create_fallback_groupings(7)

        assert groupings == [[0, 1, 2], [3, 4, 5], [6]]

    def should_handle_single_unit(self):
        groupings = create_fallback_groupings(1)

        assert groupings == [[0]]

    def should_handle_zero_units(self):
        groupings = create_fallback_groupings(0)

        assert groupings == [[]]


class DescribeFallbackGrouping:
    def should_group_units_by_size_limit(self):
        units = [
            {'content': 'A' * 500, 'type': 'paragraph', 'length': 500},
            {'content': 'B' * 400, 'type': 'paragraph', 'length': 400},
            {'content': 'C' * 300, 'type': 'paragraph', 'length': 300},
        ]

        groupings = fallback_grouping(units)

        # First two should be grouped together (900 chars < 1000 limit)
        # Third should be separate
        assert len(groupings) == 2
        assert groupings[0] == [0, 1]
        assert groupings[1] == [2]

    def should_start_new_group_after_headers(self):
        units = [
            {'content': 'Paragraph 1', 'type': 'paragraph', 'length': 11},
            {'content': '# Header', 'type': 'header', 'length': 8},
            {'content': 'Paragraph 2', 'type': 'paragraph', 'length': 11},
        ]

        groupings = fallback_grouping(units)

        # Header should start a new group
        assert len(groupings) == 2
        assert groupings[0] == [0]
        assert groupings[1] == [1, 2]

    def should_handle_oversized_single_unit(self):
        units = [
            {'content': 'A' * 1500, 'type': 'paragraph', 'length': 1500},
        ]

        groupings = fallback_grouping(units)

        # Should still create a group even if it exceeds the limit
        assert groupings == [[0]]

    def should_handle_empty_units_list(self):
        units = []

        groupings = fallback_grouping(units)

        assert groupings == []
