"""
Tests for context detection functionality.
"""

import pytest
from typing import List

from context_mixer.domain.context import Context, ContextType, ContextAnalysis
from context_mixer.domain.context_detection import (
    ContextDetector,
    ArchitecturalContextDetector,
    PlatformContextDetector,
    EnvironmentContextDetector,
    LanguageContextDetector,
    ContextDetectionEngine
)


class DescribeArchitecturalContextDetector:
    """Test architectural context detection."""

    @pytest.fixture
    def detector(self):
        """Create an architectural context detector."""
        return ArchitecturalContextDetector()

    def should_detect_gateway_context(self, detector):
        """Test detection of gateway architectural context."""
        content = "Use the Gateway pattern to isolate I/O from business logic. Gateways should not be tested."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.ARCHITECTURAL
        assert contexts[0].value == "gateway"
        assert contexts[0].confidence is not None
        assert contexts[0].confidence > 0.5

    def should_detect_service_context(self, detector):
        """Test detection of service architectural context."""
        content = "Services contain business logic and should be unit tested with mocked dependencies."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.ARCHITECTURAL
        assert contexts[0].value == "service"

    def should_detect_multiple_architectural_contexts(self, detector):
        """Test detection of multiple architectural contexts."""
        content = "Controllers handle HTTP requests and delegate to services. Services contain business logic."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 2
        context_values = [ctx.value for ctx in contexts]
        assert "controller" in context_values
        assert "service" in context_values

    def should_not_detect_contexts_in_unrelated_content(self, detector):
        """Test that no contexts are detected in unrelated content."""
        content = "This is just some random text about cooking recipes and gardening tips."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 0

    def should_return_supported_types(self, detector):
        """Test that detector returns correct supported types."""
        supported_types = detector.get_supported_types()
        
        assert supported_types == [ContextType.ARCHITECTURAL]


class DescribePlatformContextDetector:
    """Test platform context detection."""

    @pytest.fixture
    def detector(self):
        """Create a platform context detector."""
        return PlatformContextDetector()

    def should_detect_web_platform_context(self, detector):
        """Test detection of web platform context."""
        content = "Use localStorage for client-side storage in web applications."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.PLATFORM
        assert contexts[0].value == "web"

    def should_detect_mobile_platform_context(self, detector):
        """Test detection of mobile platform context."""
        content = "Use AsyncStorage for client-side storage in React Native mobile apps."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.PLATFORM
        assert contexts[0].value == "mobile"

    def should_detect_server_platform_context(self, detector):
        """Test detection of server platform context."""
        content = "Configure the Express.js server to handle API requests and database connections."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.PLATFORM
        assert contexts[0].value == "server"

    def should_return_supported_types(self, detector):
        """Test that detector returns correct supported types."""
        supported_types = detector.get_supported_types()
        
        assert supported_types == [ContextType.PLATFORM]


class DescribeEnvironmentContextDetector:
    """Test environment context detection."""

    @pytest.fixture
    def detector(self):
        """Create an environment context detector."""
        return EnvironmentContextDetector()

    def should_detect_development_environment_context(self, detector):
        """Test detection of development environment context."""
        content = "Enable debug logging in development environment for troubleshooting."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.ENVIRONMENT
        assert contexts[0].value == "development"

    def should_detect_production_environment_context(self, detector):
        """Test detection of production environment context."""
        content = "Disable verbose logging in production for performance optimization."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.ENVIRONMENT
        assert contexts[0].value == "production"

    def should_detect_staging_environment_context(self, detector):
        """Test detection of staging environment context."""
        content = "Run integration tests in the staging environment before deployment."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.ENVIRONMENT
        assert contexts[0].value == "staging"

    def should_return_supported_types(self, detector):
        """Test that detector returns correct supported types."""
        supported_types = detector.get_supported_types()
        
        assert supported_types == [ContextType.ENVIRONMENT]


class DescribeLanguageContextDetector:
    """Test language context detection."""

    @pytest.fixture
    def detector(self):
        """Create a language context detector."""
        return LanguageContextDetector()

    def should_detect_python_language_context(self, detector):
        """Test detection of Python language context."""
        content = "Use pytest for testing and pydantic for data validation in Python projects."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.LANGUAGE
        assert contexts[0].value == "python"

    def should_detect_javascript_language_context(self, detector):
        """Test detection of JavaScript language context."""
        content = "Use npm to install packages and React for building user interfaces."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.LANGUAGE
        assert contexts[0].value == "javascript"

    def should_detect_java_language_context(self, detector):
        """Test detection of Java language context."""
        content = "Use Maven for dependency management and Spring framework for enterprise applications."
        
        contexts = detector.detect_contexts(content)
        
        assert len(contexts) == 1
        assert contexts[0].type == ContextType.LANGUAGE
        assert contexts[0].value == "java"

    def should_return_supported_types(self, detector):
        """Test that detector returns correct supported types."""
        supported_types = detector.get_supported_types()
        
        assert supported_types == [ContextType.LANGUAGE]


class DescribeContextDetectionEngine:
    """Test the context detection engine."""

    @pytest.fixture
    def engine(self):
        """Create a context detection engine with default detectors."""
        return ContextDetectionEngine()

    @pytest.fixture
    def custom_engine(self):
        """Create a context detection engine with custom detectors."""
        detectors = [
            ArchitecturalContextDetector(),
            PlatformContextDetector()
        ]
        return ContextDetectionEngine(detectors)

    def should_detect_multiple_context_types(self, engine):
        """Test detection of multiple context types in single content."""
        content = """
        Use the Gateway pattern for I/O operations in Python web applications.
        Enable debug logging in development environment.
        """
        
        analysis = engine.detect_contexts(content)
        
        assert isinstance(analysis, ContextAnalysis)
        assert len(analysis.detected_contexts) >= 3  # architectural, language, platform, environment
        
        context_types = {ctx.type for ctx in analysis.detected_contexts}
        assert ContextType.ARCHITECTURAL in context_types
        assert ContextType.LANGUAGE in context_types
        assert ContextType.PLATFORM in context_types
        assert ContextType.ENVIRONMENT in context_types

    def should_remove_duplicate_contexts(self, engine):
        """Test that duplicate contexts are removed."""
        content = "Use gateways for I/O. Gateways should isolate I/O operations."
        
        analysis = engine.detect_contexts(content)
        
        # Should only have one gateway context despite multiple mentions
        gateway_contexts = [ctx for ctx in analysis.detected_contexts if ctx.value == "gateway"]
        assert len(gateway_contexts) == 1

    def should_calculate_overall_confidence(self, engine):
        """Test that overall confidence is calculated correctly."""
        content = "Use the Gateway pattern for I/O operations in Python applications."
        
        analysis = engine.detect_contexts(content)
        
        assert analysis.confidence_score >= 0.0
        assert analysis.confidence_score <= 1.0

    def should_generate_analysis_notes(self, engine):
        """Test that analysis notes are generated."""
        content = "Use the Gateway pattern for I/O operations in Python web applications."
        
        analysis = engine.detect_contexts(content)
        
        assert analysis.analysis_notes is not None
        assert len(analysis.analysis_notes) > 0

    def should_handle_empty_content(self, engine):
        """Test handling of empty content."""
        content = ""
        
        analysis = engine.detect_contexts(content)
        
        assert isinstance(analysis, ContextAnalysis)
        assert len(analysis.detected_contexts) == 0
        assert analysis.confidence_score == 0.0

    def should_handle_content_with_no_contexts(self, engine):
        """Test handling of content with no detectable contexts."""
        content = "This is just some random text about cooking and gardening."
        
        analysis = engine.detect_contexts(content)
        
        assert isinstance(analysis, ContextAnalysis)
        assert len(analysis.detected_contexts) == 0
        assert "No specific contexts detected" in analysis.analysis_notes

    def should_use_custom_detectors(self, custom_engine):
        """Test that custom detectors are used correctly."""
        supported_types = custom_engine.get_supported_types()
        
        assert ContextType.ARCHITECTURAL in supported_types
        assert ContextType.PLATFORM in supported_types
        assert ContextType.ENVIRONMENT not in supported_types  # Not included in custom engine
        assert ContextType.LANGUAGE not in supported_types  # Not included in custom engine

    def should_add_detector_dynamically(self, custom_engine):
        """Test adding a detector dynamically."""
        # Initially doesn't support environment contexts
        assert ContextType.ENVIRONMENT not in custom_engine.get_supported_types()
        
        # Add environment detector
        custom_engine.add_detector(EnvironmentContextDetector())
        
        # Now supports environment contexts
        assert ContextType.ENVIRONMENT in custom_engine.get_supported_types()

    def should_handle_detector_errors_gracefully(self, mocker):
        """Test that detector errors are handled gracefully."""
        # Create a mock detector that raises an exception
        failing_detector = mocker.MagicMock(spec=ContextDetector)
        failing_detector.detect_contexts.side_effect = Exception("Detector failed")
        failing_detector.get_detector_name.return_value = "FailingDetector"
        failing_detector.get_supported_types.return_value = [ContextType.CUSTOM]
        
        # Create engine with failing detector and a working one
        working_detector = ArchitecturalContextDetector()
        engine = ContextDetectionEngine([failing_detector, working_detector])
        
        content = "Use the Gateway pattern for I/O operations."
        
        # Should still work despite one detector failing
        analysis = engine.detect_contexts(content)
        
        assert isinstance(analysis, ContextAnalysis)
        # Should have contexts from the working detector
        gateway_contexts = [ctx for ctx in analysis.detected_contexts if ctx.value == "gateway"]
        assert len(gateway_contexts) == 1