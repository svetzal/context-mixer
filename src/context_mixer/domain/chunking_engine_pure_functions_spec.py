from context_mixer.domain.chunking_engine import (
    generate_chunk_id,
    split_content_on_blank_lines,
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

        units = split_content_on_blank_lines(content)

        assert len(units) == 1
        assert units[0]['content'] == "This is a single paragraph."
        assert units[0]['start_pos'] == 0
        assert units[0]['end_pos'] == 27

    def should_extract_multiple_paragraphs(self):
        content = "First paragraph.\n\nSecond paragraph."

        units = split_content_on_blank_lines(content)

        assert len(units) == 2
        assert units[0]['content'] == "First paragraph."
        assert units[1]['content'] == "Second paragraph."

    def should_handle_empty_content(self):
        content = ""

        units = split_content_on_blank_lines(content)

        assert len(units) == 0

    def should_skip_empty_paragraphs(self):
        content = "First paragraph.\n\n\n\nSecond paragraph."

        units = split_content_on_blank_lines(content)

        assert len(units) == 2
        assert units[0]['content'] == "First paragraph."
        assert units[1]['content'] == "Second paragraph."

    def should_extract_units_from_mixed_content(self):
        content = "# Header\n\nRegular paragraph.\n\n```\ncode block\n```"

        units = split_content_on_blank_lines(content)

        assert len(units) == 3
        assert units[0]['content'] == "# Header"
        assert units[1]['content'] == "Regular paragraph."
        assert units[2]['content'] == "```\ncode block\n```"






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
            {'content': 'A' * 500, 'length': 500},
            {'content': 'B' * 400, 'length': 400},
            {'content': 'C' * 300, 'length': 300},
        ]

        groupings = fallback_grouping(units)

        # First two should be grouped together (900 chars < 1000 limit)
        # Third should be separate
        assert len(groupings) == 2
        assert groupings[0] == [0, 1]
        assert groupings[1] == [2]

    def should_group_units_by_size_only(self):
        units = [
            {'content': 'Paragraph 1', 'length': 11},
            {'content': '# Header', 'length': 8},
            {'content': 'Paragraph 2', 'length': 11},
        ]

        groupings = fallback_grouping(units)

        # All units should be grouped together since total size (30) < 1000 limit
        assert len(groupings) == 1
        assert groupings[0] == [0, 1, 2]

    def should_handle_oversized_single_unit(self):
        units = [
            {'content': 'A' * 1500, 'length': 1500},
        ]

        groupings = fallback_grouping(units)

        # Should still create a group even if it exceeds the limit
        assert groupings == [[0]]

    def should_handle_empty_units_list(self):
        units = []

        groupings = fallback_grouping(units)

        assert groupings == []
