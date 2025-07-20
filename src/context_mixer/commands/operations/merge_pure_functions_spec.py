from unittest.mock import MagicMock

from context_mixer.commands.operations.merge import (
    format_conflict_resolutions,
    build_merge_prompt
)


class DescribeFormatConflictResolutions:
    def should_return_empty_string_when_no_conflicts(self):
        result = format_conflict_resolutions(None)
        assert result == ""

    def should_return_empty_string_when_empty_list(self):
        result = format_conflict_resolutions([])
        assert result == ""

    def should_format_single_conflict_resolution(self):
        mock_conflict = MagicMock()
        mock_conflict.description = "Test conflict description"
        mock_conflict.resolution = "Test resolution"

        result = format_conflict_resolutions([mock_conflict])

        assert "Test conflict description" in result
        assert "Test resolution" in result
        assert "Conflicts were detected" in result

    def should_format_multiple_conflict_resolutions(self):
        mock_conflict1 = MagicMock()
        mock_conflict1.description = "First conflict"
        mock_conflict1.resolution = "First resolution"

        mock_conflict2 = MagicMock()
        mock_conflict2.description = "Second conflict"
        mock_conflict2.resolution = "Second resolution"

        result = format_conflict_resolutions([mock_conflict1, mock_conflict2])

        assert "First conflict" in result
        assert "First resolution" in result
        assert "Second conflict" in result
        assert "Second resolution" in result

    def should_handle_conflicts_with_resolution_none(self):
        mock_conflict1 = MagicMock()
        mock_conflict1.description = "Resolved conflict"
        mock_conflict1.resolution = "Test resolution"

        mock_conflict2 = MagicMock()
        mock_conflict2.description = "Not a conflict"
        mock_conflict2.resolution = None
        mock_conflict2.conflicting_guidance = [
            MagicMock(content="Option A", source="existing"),
            MagicMock(content="Option B", source="new")
        ]

        result = format_conflict_resolutions([mock_conflict1, mock_conflict2])

        assert "Resolved conflict" in result
        assert "Test resolution" in result
        assert "Not a conflict" in result
        assert "This is not a conflict - both pieces of guidance are acceptable" in result
        assert "Option A (from existing)" in result
        assert "Option B (from new)" in result

    def should_handle_conflicts_with_only_resolution_none(self):
        mock_conflict = MagicMock()
        mock_conflict.description = "Test conflict"
        mock_conflict.resolution = None
        mock_conflict.conflicting_guidance = [
            MagicMock(content="Option A", source="existing"),
            MagicMock(content="Option B", source="new")
        ]

        result = format_conflict_resolutions([mock_conflict])

        assert result != ""
        assert "Test conflict" in result
        assert "This is not a conflict - both pieces of guidance are acceptable" in result
        assert "Option A (from existing)" in result
        assert "Option B (from new)" in result


class DescribeBuildMergePrompt:
    def should_build_basic_merge_prompt_without_conflicts(self):
        existing_content = "Existing document content"
        new_content = "New document content"

        result = build_merge_prompt(existing_content, new_content)

        assert "merge two documents" in result
        assert existing_content in result
        assert new_content in result
        assert "Document 1 (Existing Content)" in result
        assert "Document 2 (New Content)" in result
        assert "without any additional commentary" in result

    def should_include_conflict_resolutions_when_provided(self):
        existing_content = "Existing content"
        new_content = "New content"

        mock_conflict = MagicMock()
        mock_conflict.description = "Test conflict"
        mock_conflict.resolution = "Test resolution"

        result = build_merge_prompt(existing_content, new_content, [mock_conflict])

        assert existing_content in result
        assert new_content in result
        assert "Test conflict" in result
        assert "Test resolution" in result
        assert "Conflicts were detected" in result

    def should_not_include_conflict_section_when_no_conflicts(self):
        existing_content = "Existing content"
        new_content = "New content"

        result = build_merge_prompt(existing_content, new_content, None)

        assert "Conflicts were detected" not in result

    def should_not_include_conflict_section_when_empty_conflicts(self):
        existing_content = "Existing content"
        new_content = "New content"

        result = build_merge_prompt(existing_content, new_content, [])

        assert "Conflicts were detected" not in result

    def should_handle_multiline_content(self):
        existing_content = "Line 1\nLine 2\nLine 3"
        new_content = "New Line 1\nNew Line 2"

        result = build_merge_prompt(existing_content, new_content)

        assert "Line 1\nLine 2\nLine 3" in result
        assert "New Line 1\nNew Line 2" in result

    def should_include_all_required_sections(self):
        existing_content = "Test existing"
        new_content = "Test new"

        result = build_merge_prompt(existing_content, new_content)

        # Check for key sections
        assert "merge two documents" in result
        assert "All detail and unique information" in result
        assert "Duplicate information appears only once" in result
        assert "well-structured and coherent" in result
        assert "Related information is grouped together" in result
        assert "without any additional commentary" in result

    def should_structure_prompt_parts_correctly(self):
        existing_content = "Test existing"
        new_content = "Test new"

        mock_conflict = MagicMock()
        mock_conflict.description = "Test conflict"
        mock_conflict.resolution = "Test resolution"

        result = build_merge_prompt(existing_content, new_content, [mock_conflict])

        # Should have base prompt, conflict info, and conclusion separated by double newlines
        parts = result.split('\n\n')
        assert len(parts) >= 3  # At least base, conflict, conclusion

        # Conflict info should be between base and conclusion
        conflict_found = False
        for part in parts:
            if "Conflicts were detected" in part:
                conflict_found = True
                break
        assert conflict_found
