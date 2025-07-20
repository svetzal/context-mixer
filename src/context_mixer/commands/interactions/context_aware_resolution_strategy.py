from typing import List, Optional

from rich.console import Console

from context_mixer.domain.conflict import Conflict
from context_mixer.domain.context import Context, ContextType
from context_mixer.domain.context_aware_prompts import ContextAwarePromptBuilder
from context_mixer.domain.context_detection import ContextDetectionEngine
from context_mixer.gateways.llm import LLMGateway
from .conflict_resolution_strategies import ConflictResolutionStrategy


class ContextAwareResolutionStrategy(ConflictResolutionStrategy):
    """
    Context-aware conflict resolution strategy.
    
    This strategy analyzes conflicts using context information to determine
    if they are genuine conflicts or just rules that apply to different contexts.
    It can auto-resolve conflicts that are clearly not conflicting due to
    different contexts, and provide better guidance for genuine conflicts.
    """
    
    def __init__(self, 
                 llm_gateway: Optional[LLMGateway] = None,
                 context_engine: Optional[ContextDetectionEngine] = None,
                 prompt_builder: Optional[ContextAwarePromptBuilder] = None,
                 auto_resolve_threshold: float = 0.8):
        """
        Initialize the context-aware resolution strategy.
        
        Args:
            llm_gateway: LLM gateway for generating resolutions
            context_engine: Context detection engine
            prompt_builder: Context-aware prompt builder
            auto_resolve_threshold: Confidence threshold for auto-resolution (0.0-1.0)
        """
        self.llm_gateway = llm_gateway
        self.context_engine = context_engine or ContextDetectionEngine()
        self.prompt_builder = prompt_builder or ContextAwarePromptBuilder(self.context_engine)
        self.auto_resolve_threshold = auto_resolve_threshold
    
    def resolve_conflicts(self, conflicts: List[Conflict], console: Optional[Console] = None) -> List[Conflict]:
        """
        Resolve conflicts using context-aware analysis.
        
        Args:
            conflicts: List of conflicts to resolve
            console: Rich console for output
            
        Returns:
            List of resolved conflicts
        """
        if not conflicts:
            return []
        
        resolved_conflicts = []
        
        for conflict in conflicts:
            # Analyze contexts for each conflicting guidance
            self._analyze_conflict_contexts(conflict)
            
            # Check if contexts suggest this is not a real conflict
            if self._contexts_suggest_no_conflict(conflict):
                # Auto-resolve with context-aware explanation
                conflict.resolution = self._generate_context_aware_resolution(conflict)
                conflict.context_analysis = self._generate_context_analysis(conflict)
                
                if console:
                    console.print(f"[green]Auto-resolved conflict based on context analysis:[/green]")
                    console.print(f"  {conflict.description}")
                    console.print(f"  Resolution: {conflict.resolution}")
                    console.print(f"  Analysis: {conflict.context_analysis}")
                    console.print()
            else:
                # Present context-aware options to user
                if console:
                    conflict = self._resolve_with_context_awareness(conflict, console)
                else:
                    # If no console, try LLM-based resolution
                    if self.llm_gateway:
                        conflict.resolution = self._generate_llm_resolution(conflict)
                        conflict.context_analysis = self._generate_context_analysis(conflict)
            
            resolved_conflicts.append(conflict)
        
        return resolved_conflicts
    
    def _analyze_conflict_contexts(self, conflict: Conflict):
        """
        Analyze and populate context information for conflicting guidance.
        
        Args:
            conflict: The conflict to analyze
        """
        for guidance in conflict.conflicting_guidance:
            if not guidance.contexts:  # Only analyze if contexts not already set
                analysis = self.context_engine.detect_contexts(guidance.content)
                guidance.contexts = analysis.detected_contexts
                guidance.confidence = analysis.confidence_score
    
    def _contexts_suggest_no_conflict(self, conflict: Conflict) -> bool:
        """
        Determine if the contexts suggest this is not a real conflict.
        
        Args:
            conflict: The conflict to analyze
            
        Returns:
            True if contexts suggest no real conflict
        """
        if len(conflict.conflicting_guidance) < 2:
            return False
        
        guidance1 = conflict.conflicting_guidance[0]
        guidance2 = conflict.conflicting_guidance[1]
        
        # If either guidance has no contexts, we can't make a determination
        if not guidance1.contexts or not guidance2.contexts:
            return False
        
        # Check if contexts are completely different
        contexts1 = set((ctx.type, ctx.value) for ctx in guidance1.contexts)
        contexts2 = set((ctx.type, ctx.value) for ctx in guidance2.contexts)
        
        # If there's no overlap in contexts, they likely don't conflict
        if not contexts1.intersection(contexts2):
            # Additional check: ensure both have reasonable confidence
            min_confidence = min(
                guidance1.confidence or 0.0,
                guidance2.confidence or 0.0
            )
            return min_confidence >= self.auto_resolve_threshold
        
        # Check for specific non-conflicting context patterns
        return self._check_specific_non_conflict_patterns(guidance1.contexts, guidance2.contexts)
    
    def _check_specific_non_conflict_patterns(self, contexts1: List[Context], contexts2: List[Context]) -> bool:
        """
        Check for specific patterns that indicate non-conflicting contexts.
        
        Args:
            contexts1: Contexts from first guidance
            contexts2: Contexts from second guidance
            
        Returns:
            True if specific non-conflict patterns are detected
        """
        # Different platforms
        platforms1 = {ctx.value for ctx in contexts1 if ctx.type == ContextType.PLATFORM}
        platforms2 = {ctx.value for ctx in contexts2 if ctx.type == ContextType.PLATFORM}
        if platforms1 and platforms2 and not platforms1.intersection(platforms2):
            return True
        
        # Different environments
        envs1 = {ctx.value for ctx in contexts1 if ctx.type == ContextType.ENVIRONMENT}
        envs2 = {ctx.value for ctx in contexts2 if ctx.type == ContextType.ENVIRONMENT}
        if envs1 and envs2 and not envs1.intersection(envs2):
            return True
        
        # Different languages
        langs1 = {ctx.value for ctx in contexts1 if ctx.type == ContextType.LANGUAGE}
        langs2 = {ctx.value for ctx in contexts2 if ctx.type == ContextType.LANGUAGE}
        if langs1 and langs2 and not langs1.intersection(langs2):
            return True
        
        # Different architectural components
        arch1 = {ctx.value for ctx in contexts1 if ctx.type == ContextType.ARCHITECTURAL}
        arch2 = {ctx.value for ctx in contexts2 if ctx.type == ContextType.ARCHITECTURAL}
        if arch1 and arch2 and not arch1.intersection(arch2):
            return True
        
        return False
    
    def _generate_context_aware_resolution(self, conflict: Conflict) -> str:
        """
        Generate a context-aware resolution for the conflict.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            Resolution text
        """
        guidance1 = conflict.conflicting_guidance[0]
        guidance2 = conflict.conflicting_guidance[1]
        
        # Create a resolution that preserves both pieces of guidance with their contexts
        contexts1_str = ", ".join(str(ctx) for ctx in guidance1.contexts)
        contexts2_str = ", ".join(str(ctx) for ctx in guidance2.contexts)
        
        resolution = f"""Both pieces of guidance are valid in their respective contexts:

1. {guidance1.content.strip()}
   Context: {contexts1_str}

2. {guidance2.content.strip()}
   Context: {contexts2_str}

These rules apply to different contexts and should both be followed in their appropriate situations."""
        
        return resolution
    
    def _generate_context_analysis(self, conflict: Conflict) -> str:
        """
        Generate context analysis explanation.
        
        Args:
            conflict: The conflict to analyze
            
        Returns:
            Context analysis text
        """
        guidance1 = conflict.conflicting_guidance[0]
        guidance2 = conflict.conflicting_guidance[1]
        
        if not guidance1.contexts or not guidance2.contexts:
            return "Context analysis: Insufficient context information to determine relationship."
        
        contexts1 = set((ctx.type, ctx.value) for ctx in guidance1.contexts)
        contexts2 = set((ctx.type, ctx.value) for ctx in guidance2.contexts)
        
        if not contexts1.intersection(contexts2):
            return "Context analysis: These rules apply to completely different contexts and do not conflict."
        
        overlap = contexts1.intersection(contexts2)
        different = contexts1.symmetric_difference(contexts2)
        
        analysis = f"Context analysis: Rules have {len(overlap)} overlapping contexts and {len(different)} different contexts. "
        
        if len(different) > len(overlap):
            analysis += "The differences suggest these may be complementary rather than conflicting rules."
        else:
            analysis += "The overlap suggests these rules may genuinely conflict and require resolution."
        
        return analysis
    
    def _resolve_with_context_awareness(self, conflict: Conflict, console: Console) -> Conflict:
        """
        Resolve conflict with user interaction, providing context information.
        
        Args:
            conflict: The conflict to resolve
            console: Rich console for interaction
            
        Returns:
            Resolved conflict
        """
        console.print(f"[yellow]Conflict detected:[/yellow] {conflict.description}")
        console.print()
        
        # Display conflicting guidance with context information
        for i, guidance in enumerate(conflict.conflicting_guidance, 1):
            console.print(f"[cyan]Option {i}:[/cyan] {guidance.content}")
            if guidance.contexts:
                contexts_str = ", ".join(str(ctx) for ctx in guidance.contexts)
                console.print(f"  [dim]Contexts: {contexts_str}[/dim]")
            console.print()
        
        # Show context analysis if available
        if conflict.context_analysis:
            console.print(f"[blue]Context Analysis:[/blue] {conflict.context_analysis}")
            console.print()
        
        # Provide resolution options
        console.print("Resolution options:")
        console.print("1. Keep both (they apply to different contexts)")
        console.print("2. Choose option 1")
        console.print("3. Choose option 2")
        console.print("4. Provide custom resolution")
        
        while True:
            choice = console.input("Select option (1-4): ").strip()
            
            if choice == "1":
                conflict.resolution = self._generate_context_aware_resolution(conflict)
                break
            elif choice == "2":
                conflict.resolution = conflict.conflicting_guidance[0].content
                break
            elif choice == "3":
                conflict.resolution = conflict.conflicting_guidance[1].content
                break
            elif choice == "4":
                custom_resolution = console.input("Enter custom resolution: ").strip()
                if custom_resolution:
                    conflict.resolution = custom_resolution
                    break
            else:
                console.print("[red]Invalid choice. Please select 1-4.[/red]")
        
        return conflict
    
    def _generate_llm_resolution(self, conflict: Conflict) -> str:
        """
        Generate LLM-based resolution using context information.
        
        Args:
            conflict: The conflict to resolve
            
        Returns:
            LLM-generated resolution
        """
        if not self.llm_gateway:
            return "Unable to generate resolution: No LLM gateway available."
        
        # Use the context-aware prompt builder to create a resolution prompt
        guidance1 = conflict.conflicting_guidance[0]
        guidance2 = conflict.conflicting_guidance[1]
        
        prompt = f"""
        Resolve the following conflict by considering the contexts in which each piece of guidance applies:
        
        Conflict: {conflict.description}
        
        Guidance 1: {guidance1.content}
        Contexts: {', '.join(str(ctx) for ctx in guidance1.contexts) if guidance1.contexts else 'None detected'}
        
        Guidance 2: {guidance2.content}
        Contexts: {', '.join(str(ctx) for ctx in guidance2.contexts) if guidance2.contexts else 'None detected'}
        
        Context Analysis: {conflict.context_analysis or 'Not available'}
        
        Please provide a resolution that:
        1. Considers the different contexts where each rule applies
        2. Preserves both rules if they apply to different contexts
        3. Provides clear guidance on when to apply each rule
        4. Resolves any genuine conflicts if they exist
        
        Resolution:
        """
        
        try:
            from mojentic.llm import LLMMessage
            messages = [LLMMessage(content=prompt)]
            response = self.llm_gateway.generate(messages)
            return response.strip()
        except Exception as e:
            return f"Error generating LLM resolution: {str(e)}"
    
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return "context_aware"