"""
Tests for timing utilities.
"""

import pytest
import time
from unittest.mock import patch

from context_mixer.utils.timing import (
    TimingResult,
    TimingCollector,
    time_operation,
    format_duration
)


class DescribeTimingResult:
    def should_create_timing_result_with_all_fields(self):
        result = TimingResult(
            operation_name="test_operation",
            duration_seconds=1.5,
            start_time=100.0,
            end_time=101.5
        )
        
        assert result.operation_name == "test_operation"
        assert result.duration_seconds == 1.5
        assert result.start_time == 100.0
        assert result.end_time == 101.5


class DescribeTimingCollector:
    @pytest.fixture
    def collector(self):
        return TimingCollector()
    
    @pytest.fixture
    def sample_results(self):
        return [
            TimingResult("operation1", 1.0, 100.0, 101.0),
            TimingResult("operation2", 2.5, 102.0, 104.5),
            TimingResult("operation1", 0.5, 105.0, 105.5)  # Duplicate name
        ]
    
    def should_start_with_empty_results(self, collector):
        assert len(collector.results) == 0
        assert collector.get_total_duration() == 0.0
    
    def should_add_timing_results(self, collector, sample_results):
        for result in sample_results:
            collector.add_result(result)
        
        assert len(collector.results) == 3
        assert collector.results == sample_results
    
    def should_calculate_total_duration(self, collector, sample_results):
        for result in sample_results:
            collector.add_result(result)
        
        expected_total = 1.0 + 2.5 + 0.5
        assert collector.get_total_duration() == expected_total
    
    def should_get_operation_duration_for_existing_operation(self, collector, sample_results):
        for result in sample_results:
            collector.add_result(result)
        
        # Should return the first matching operation
        duration = collector.get_operation_duration("operation1")
        assert duration == 1.0
        
        duration = collector.get_operation_duration("operation2")
        assert duration == 2.5
    
    def should_return_none_for_nonexistent_operation(self, collector):
        duration = collector.get_operation_duration("nonexistent")
        assert duration is None
    
    def should_get_summary_of_all_operations(self, collector, sample_results):
        for result in sample_results:
            collector.add_result(result)
        
        summary = collector.get_summary()
        expected_summary = {
            "operation1": 1.0,  # First occurrence
            "operation2": 2.5,
            "operation1": 0.5   # This will overwrite the first one in dict
        }
        
        # Note: Dict will have the last value for duplicate keys
        assert "operation1" in summary
        assert "operation2" in summary
        assert summary["operation2"] == 2.5


class DescribeTimeOperation:
    def should_time_operation_and_return_result(self):
        with time_operation("test_op") as timing:
            time.sleep(0.01)  # Small delay for measurable time
        
        assert timing.operation_name == "test_op"
        assert timing.duration_seconds > 0.0
        assert timing.start_time > 0.0
        assert timing.end_time > timing.start_time
        assert timing.duration_seconds == timing.end_time - timing.start_time
    
    def should_add_result_to_collector_when_provided(self):
        collector = TimingCollector()
        
        with time_operation("test_op", collector) as timing:
            time.sleep(0.01)
        
        assert len(collector.results) == 1
        assert collector.results[0] == timing
        assert collector.results[0].operation_name == "test_op"
    
    def should_work_without_collector(self):
        with time_operation("test_op") as timing:
            time.sleep(0.01)
        
        assert timing.operation_name == "test_op"
        assert timing.duration_seconds > 0.0
    
    def should_handle_exceptions_and_still_record_timing(self):
        collector = TimingCollector()
        
        with pytest.raises(ValueError):
            with time_operation("failing_op", collector) as timing:
                time.sleep(0.01)
                raise ValueError("Test exception")
        
        # Should still record the timing even when exception occurs
        assert len(collector.results) == 1
        assert collector.results[0].operation_name == "failing_op"
        assert collector.results[0].duration_seconds > 0.0


class DescribeFormatDuration:
    def should_format_milliseconds_for_sub_second_durations(self):
        assert format_duration(0.001) == "1ms"
        assert format_duration(0.123) == "123ms"
        assert format_duration(0.999) == "999ms"
    
    def should_format_seconds_for_short_durations(self):
        assert format_duration(1.0) == "1.00s"
        assert format_duration(1.23) == "1.23s"
        assert format_duration(59.99) == "59.99s"
    
    def should_format_minutes_and_seconds_for_long_durations(self):
        assert format_duration(60.0) == "1m 0.0s"
        assert format_duration(61.5) == "1m 1.5s"
        assert format_duration(125.7) == "2m 5.7s"
        assert format_duration(3661.2) == "61m 1.2s"
    
    def should_handle_zero_duration(self):
        assert format_duration(0.0) == "0ms"