from abc import ABC, abstractmethod
from typing import List, Set

from .context import Context, ContextType, ContextAnalysis


class ContextDetector(ABC):
    """
    Abstract base class for context detectors.

    Context detectors analyze content and identify contexts where
    rules or guidance might apply, helping to reduce false positive
    conflict detection.
    """

    @abstractmethod
    def detect_contexts(self, content: str) -> List[Context]:
        """
        Detect contexts in the given content.

        Args:
            content: The text content to analyze

        Returns:
            List of detected contexts
        """
        pass

    @abstractmethod
    def get_supported_types(self) -> List[ContextType]:
        """
        Get the context types this detector supports.

        Returns:
            List of supported context types
        """
        pass

    def get_detector_name(self) -> str:
        """Get the name of this detector."""
        return self.__class__.__name__


class ArchitecturalContextDetector(ContextDetector):
    """
    Detects architectural contexts in content.

    This detector identifies architectural patterns, components,
    and layers mentioned in the content.
    """

    def __init__(self):
        """Initialize the architectural context detector."""
        # Architectural patterns and components
        self.architectural_patterns = {
            'gateway': ['gateway pattern', 'gateway', 'gateways', 'i/o boundary', 'io boundary'],
            'repository': ['repository pattern', 'repository', 'repositories', 'data access', 'persistence layer'],
            'service': ['service layer', 'domain service', 'business service', 'application service', 'services contain'],
            'controller': ['controller', 'controllers', 'api controller', 'web controller'],
            'model': ['domain model', 'entity', 'entities', 'data model'],
            'view': ['view', 'views', 'presentation layer', 'user interface'],
            'middleware': ['middleware', 'interceptor', 'filter'],
            'adapter': ['adapter pattern', 'adapter', 'adapters'],
            'factory': ['factory pattern', 'factory', 'factories', 'builder pattern'],
            'observer': ['observer pattern', 'observer', 'listener', 'subscriber', 'event handler'],
            'strategy': ['strategy pattern', 'algorithm', 'policy'],
            'decorator': ['decorator pattern', 'decorator', 'enhancement'],
        }

    def detect_contexts(self, content: str) -> List[Context]:
        """
        Detect architectural contexts in content.

        Args:
            content: The text content to analyze

        Returns:
            List of detected architectural contexts
        """
        contexts = []
        content_lower = content.lower()

        for pattern_name, keywords in self.architectural_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Calculate confidence based on keyword specificity and frequency
                    confidence = self._calculate_confidence(content_lower, keyword, keywords)

                    context = Context(
                        type=ContextType.ARCHITECTURAL,
                        value=pattern_name,
                        description=f"Architectural pattern: {pattern_name}",
                        confidence=confidence
                    )

                    # Avoid duplicates
                    if context not in contexts:
                        contexts.append(context)
                    break  # Found this pattern, move to next

        return contexts

    def _calculate_confidence(self, content: str, keyword: str, all_keywords: List[str]) -> float:
        """
        Calculate confidence score for a detected context.

        Args:
            content: The content being analyzed
            keyword: The specific keyword that matched
            all_keywords: All keywords for this pattern

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.5

        # Higher confidence for more specific keywords
        if len(keyword.split()) > 1:  # Multi-word keywords are more specific
            confidence += 0.2

        # Higher confidence for exact matches
        if keyword == keyword.strip():
            confidence += 0.1

        # Higher confidence if multiple related keywords are found
        found_keywords = sum(1 for kw in all_keywords if kw in content)
        if found_keywords > 1:
            confidence += min(0.2, found_keywords * 0.05)

        return min(1.0, confidence)

    def get_supported_types(self) -> List[ContextType]:
        """Get supported context types."""
        return [ContextType.ARCHITECTURAL]


class PlatformContextDetector(ContextDetector):
    """
    Detects platform-specific contexts in content.

    This detector identifies platform-specific rules and guidance.
    """

    def __init__(self):
        """Initialize the platform context detector."""
        self.platform_patterns = {
            'web': ['web application', 'web app', 'browser', 'html', 'css', 'document object model', 'localstorage', 'sessionstorage'],
            'mobile': ['mobile', 'ios', 'android', 'react native', 'flutter', 'asyncstorage', 'mobile app'],
            'desktop': ['desktop', 'electron', 'tauri', 'native app', 'windows app', 'macos app', 'linux app'],
            'server': ['server', 'backend', 'api server', 'database server', 'node.js', 'express', 'fastapi'],
            'cloud': ['cloud', 'aws', 'azure', 'gcp', 'serverless', 'lambda', 'cloud functions'],
        }

    def detect_contexts(self, content: str) -> List[Context]:
        """Detect platform contexts in content."""
        contexts = []
        content_lower = content.lower()

        for platform_name, keywords in self.platform_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    confidence = self._calculate_confidence(content_lower, keyword, keywords)

                    context = Context(
                        type=ContextType.PLATFORM,
                        value=platform_name,
                        description=f"Platform: {platform_name}",
                        confidence=confidence
                    )

                    if context not in contexts:
                        contexts.append(context)
                    break

        return contexts

    def _calculate_confidence(self, content: str, keyword: str, all_keywords: List[str]) -> float:
        """Calculate confidence score for platform detection."""
        confidence = 0.6  # Platform keywords tend to be more specific

        # Higher confidence for technology-specific terms
        tech_terms = ['localstorage', 'asyncstorage', 'react native', 'node.js']
        if keyword in tech_terms:
            confidence += 0.2

        # Multiple related keywords increase confidence
        found_keywords = sum(1 for kw in all_keywords if kw in content)
        if found_keywords > 1:
            confidence += min(0.2, found_keywords * 0.05)

        return min(1.0, confidence)

    def get_supported_types(self) -> List[ContextType]:
        """Get supported context types."""
        return [ContextType.PLATFORM]


class EnvironmentContextDetector(ContextDetector):
    """
    Detects environment-specific contexts in content.

    This detector identifies environment-specific rules like
    development, staging, production settings.
    """

    def __init__(self):
        """Initialize the environment context detector."""
        self.environment_patterns = {
            'development': ['development environment', 'dev environment', 'local environment', 'debug', 'debugging'],
            'staging': ['staging environment', 'staging', 'pre-production', 'qa environment'],
            'production': ['production environment', 'production', 'prod', 'live environment'],
        }

    def detect_contexts(self, content: str) -> List[Context]:
        """Detect environment contexts in content."""
        contexts = []
        content_lower = content.lower()

        for env_name, keywords in self.environment_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    confidence = 0.7  # Environment terms are usually quite specific

                    context = Context(
                        type=ContextType.ENVIRONMENT,
                        value=env_name,
                        description=f"Environment: {env_name}",
                        confidence=confidence
                    )

                    if context not in contexts:
                        contexts.append(context)
                    break

        return contexts

    def get_supported_types(self) -> List[ContextType]:
        """Get supported context types."""
        return [ContextType.ENVIRONMENT]


class LanguageContextDetector(ContextDetector):
    """
    Detects programming language-specific contexts in content.

    This detector identifies language-specific rules and conventions.
    """

    def __init__(self):
        """Initialize the language context detector."""
        self.language_patterns = {
            'python': ['python', 'pip', 'pytest', 'django', 'flask', 'pydantic', '__init__', '.py'],
            'javascript': ['javascript', 'npm', 'react', 'vue', 'angular', 'typescript', '.js'],
            'java': ['java language', 'maven', 'gradle', 'spring framework', 'junit', 'java class'],
            'csharp': ['c#', 'csharp', '.net', 'nuget', 'visual studio', 'asp.net'],
            'go': ['golang', 'go mod', 'goroutine', 'channel', 'go language'],
            'rust': ['rust language', 'cargo', 'crate', 'trait', 'impl'],
        }

    def detect_contexts(self, content: str) -> List[Context]:
        """Detect language contexts in content."""
        contexts = []
        content_lower = content.lower()

        for lang_name, keywords in self.language_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    confidence = 0.8  # Language-specific terms are usually very specific

                    context = Context(
                        type=ContextType.LANGUAGE,
                        value=lang_name,
                        description=f"Programming language: {lang_name}",
                        confidence=confidence
                    )

                    if context not in contexts:
                        contexts.append(context)
                    break

        return contexts

    def get_supported_types(self) -> List[ContextType]:
        """Get supported context types."""
        return [ContextType.LANGUAGE]


class ContextDetectionEngine:
    """
    Engine that coordinates multiple context detectors.

    This class manages a collection of context detectors and
    provides a unified interface for context detection.
    """

    def __init__(self, detectors: List[ContextDetector] = None):
        """
        Initialize the context detection engine.

        Args:
            detectors: List of context detectors to use
        """
        if detectors is None:
            # Default detectors
            self.detectors = [
                ArchitecturalContextDetector(),
                PlatformContextDetector(),
                EnvironmentContextDetector(),
                LanguageContextDetector(),
            ]
        else:
            self.detectors = detectors

    def detect_contexts(self, content: str) -> ContextAnalysis:
        """
        Detect all contexts in the given content.

        Args:
            content: The text content to analyze

        Returns:
            ContextAnalysis with detected contexts and analysis
        """
        all_contexts = []
        confidence_scores = []

        # Run all detectors
        for detector in self.detectors:
            try:
                contexts = detector.detect_contexts(content)
                all_contexts.extend(contexts)

                # Collect confidence scores
                for context in contexts:
                    if context.confidence is not None:
                        confidence_scores.append(context.confidence)

            except Exception as e:
                # Log error but continue with other detectors
                print(f"Error in {detector.get_detector_name()}: {e}")

        # Remove duplicates while preserving order
        unique_contexts = []
        seen = set()
        for context in all_contexts:
            if context not in seen:
                unique_contexts.append(context)
                seen.add(context)

        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        # Generate analysis notes
        analysis_notes = self._generate_analysis_notes(unique_contexts)

        return ContextAnalysis(
            detected_contexts=unique_contexts,
            confidence_score=min(1.0, overall_confidence),
            analysis_notes=analysis_notes,
            potential_conflicts=[]  # Could be enhanced to detect conflicting contexts
        )

    def _generate_analysis_notes(self, contexts: List[Context]) -> str:
        """Generate human-readable analysis notes."""
        if not contexts:
            return "No specific contexts detected."

        context_types = {}
        for context in contexts:
            if context.type not in context_types:
                context_types[context.type] = []
            context_types[context.type].append(context.value)

        notes = []
        for context_type, values in context_types.items():
            notes.append(f"{context_type.value}: {', '.join(values)}")

        return "; ".join(notes)

    def add_detector(self, detector: ContextDetector):
        """Add a new context detector."""
        self.detectors.append(detector)

    def get_supported_types(self) -> Set[ContextType]:
        """Get all supported context types across all detectors."""
        supported_types = set()
        for detector in self.detectors:
            supported_types.update(detector.get_supported_types())
        return supported_types
