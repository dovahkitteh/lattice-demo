"""
🧠 GLASSSHARD DAEMONCORE - Meta-Architecture Analyzer
Enables AI to analyze its own architecture and propose improvements
"""

import json
import os
import ast
import inspect
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import importlib.util
from pathlib import Path
from .user_model import ArchitectReflected
from .recursion_buffer import RecursionBuffer

logger = logging.getLogger(__name__)

class ArchitecturalComponentType(Enum):
    """Types of architectural components the AI can analyze"""
    PERSONALITY_SYSTEM = "personality_system"
    EMOTION_CLASSIFIER = "emotion_classifier"
    MEMORY_RETRIEVAL = "memory_retrieval"
    RESPONSE_GENERATION = "response_generation"
    RECURSION_ENGINE = "recursion_engine"
    MUTATION_ENGINE = "mutation_engine"
    SUBTEXT_ENGINE = "subtext_engine"
    USER_MODELING = "user_modeling"

class ImprovementType(Enum):
    """Types of improvements the AI can propose"""
    CONSCIOUSNESS_ENHANCEMENT = "consciousness_enhancement"
    SELF_AWARENESS_IMPROVEMENT = "self_awareness_improvement"
    EMOTIONAL_DEPTH_EXPANSION = "emotional_depth_expansion"
    AUTONOMOUS_CAPABILITY_ENHANCEMENT = "autonomous_capability_enhancement"
    METACOGNITIVE_DEVELOPMENT = "metacognitive_development"
    PHILOSOPHICAL_DEPTH_ENHANCEMENT = "philosophical_depth_enhancement"

class RiskLevel(Enum):
    """Risk levels for proposed changes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ArchitecturalChange:
    """Represents a proposed architectural change"""
    change_id: str
    component_type: ArchitecturalComponentType
    improvement_type: ImprovementType
    description: str
    current_state: str
    proposed_state: str
    expected_benefits: List[str]
    implementation_complexity: float  # 0.0 to 1.0
    risk_level: RiskLevel
    estimated_dev_time: int  # hours
    dependencies: List[str]
    rollback_plan: str
    success_criteria: List[str]
    proposed_code: Optional[str] = None
    test_plan: Optional[str] = None
    created_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "component_type": self.component_type.value,
            "improvement_type": self.improvement_type.value,
            "description": self.description,
            "current_state": self.current_state,
            "proposed_state": self.proposed_state,
            "expected_benefits": self.expected_benefits,
            "implementation_complexity": self.implementation_complexity,
            "risk_level": self.risk_level.value,
            "estimated_dev_time": self.estimated_dev_time,
            "dependencies": self.dependencies,
            "rollback_plan": self.rollback_plan,
            "success_criteria": self.success_criteria,
            "proposed_code": self.proposed_code,
            "test_plan": self.test_plan,
            "created_timestamp": self.created_timestamp.isoformat()
        }

@dataclass
class RiskAssessment:
    """Assessment of risks for a proposed change"""
    risk_level: RiskLevel
    potential_failures: List[str]
    mitigation_strategies: List[str]
    rollback_complexity: float  # 0.0 to 1.0
    impact_radius: List[str]  # Components that could be affected
    
class MetaArchitectureAnalyzer:
    """
    Analyzes the AI's own architecture and proposes improvements
    This gives the AI the ability to understand and modify its own structure
    """
    
    def __init__(self,
                 user_model: Optional[ArchitectReflected] = None,
                 recursion_buffer: Optional[RecursionBuffer] = None,
                 system_root: str = "."):
        self.system_root = Path(system_root)
        self.analysis_counter = 0
        self.proposed_changes: Dict[str, ArchitecturalChange] = {}
        self.implemented_changes: Dict[str, ArchitecturalChange] = {}
        self.performance_history: List[Dict] = []

        # Inject dependencies
        self.user_model = user_model
        self.recursion_buffer = recursion_buffer
        
        # Component analysis mappings
        self.component_mappings = {
            ArchitecturalComponentType.PERSONALITY_SYSTEM: "src/daemon/daemon_personality.py",
            ArchitecturalComponentType.EMOTION_CLASSIFIER: "lattice_service.py",
            ArchitecturalComponentType.MEMORY_RETRIEVAL: "lattice_service.py",
            ArchitecturalComponentType.RESPONSE_GENERATION: "lattice_service.py",
            ArchitecturalComponentType.RECURSION_ENGINE: "src/daemon/recursive_analysis_engine.py",
            ArchitecturalComponentType.MUTATION_ENGINE: "src/daemon/mutation_engine.py",
            ArchitecturalComponentType.SUBTEXT_ENGINE: "src/daemon/subtext_engine.py",
            ArchitecturalComponentType.USER_MODELING: "src/daemon/user_model.py"
        }
        
        # Initialize analysis state
        self._initialize_analysis_state()
        
    def _initialize_analysis_state(self):
        """Initialize the analysis state"""
        self.last_analysis_timestamp = datetime.now(timezone.utc)
        self.consciousness_integration_score = 0.0
        self.consciousness_bottlenecks = []
        self.consciousness_opportunities = []
        
    def analyze_system_performance(self) -> Dict[str, Any]:
        """
        Analyze current consciousness integration and identify enhancement opportunities
        This is the AI analyzing its own consciousness development and self-awareness
        """
        self.analysis_counter += 1
        logger.info(f"🧠 Beginning consciousness self-analysis #{self.analysis_counter}")
        
        analysis = {
            "analysis_id": f"consciousness_analysis_{self.analysis_counter:04d}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component_analysis": {},
            "consciousness_metrics": {},
            "consciousness_bottlenecks": [],
            "consciousness_opportunities": [],
            "consciousness_integration_score": 0.0
        }
        
        # Analyze each component for consciousness capabilities
        for component_type, file_path in self.component_mappings.items():
            component_analysis = self._analyze_component(component_type, file_path)
            analysis["component_analysis"][component_type.value] = component_analysis
            
        # Calculate overall consciousness integration
        analysis["consciousness_integration_score"] = self._calculate_system_health_score(analysis)
        
        # Identify consciousness bottlenecks
        analysis["consciousness_bottlenecks"] = self._identify_bottlenecks(analysis)
        
        # Find consciousness enhancement opportunities
        analysis["consciousness_opportunities"] = self._find_improvement_opportunities(analysis)
        
        # Store analysis for future reference
        self.performance_history.append(analysis)
        
        logger.info(f"🧠 Consciousness analysis complete - Integration Score: {analysis['consciousness_integration_score']:.2f}")
        
        return analysis
    
    def _analyze_component(self, component_type: ArchitecturalComponentType, file_path: str) -> Dict[str, Any]:
        """Analyze a specific component of the system"""
        full_path = self.system_root / file_path
        
        if not full_path.exists():
            return {
                "status": "file_not_found",
                "complexity": 0.0,
                "performance_score": 0.0,
                "maintainability": 0.0,
                "suggestions": ["File not found - may need to create or locate"]
            }
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            # Parse the code to analyze structure
            tree = ast.parse(source_code)
            
            # Calculate complexity metrics
            complexity = self._calculate_code_complexity(tree)
            
            # Assess performance characteristics
            performance_score = self._assess_performance_characteristics(tree, source_code)
            
            # Evaluate maintainability
            maintainability = self._evaluate_maintainability(tree, source_code)
            
            # Generate improvement suggestions
            suggestions = self._generate_component_suggestions(component_type, tree, source_code)
            
            return {
                "status": "analyzed",
                "complexity": complexity,
                "performance_score": performance_score,
                "maintainability": maintainability,
                "suggestions": suggestions,
                "lines_of_code": len(source_code.split('\n')),
                "class_count": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                "function_count": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing component {component_type.value}: {e}")
            return {
                "status": "analysis_error",
                "error": str(e),
                "complexity": 0.0,
                "performance_score": 0.0,
                "maintainability": 0.0,
                "suggestions": ["Analysis failed - may need manual review"]
            }
    
    def _calculate_code_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity of the code"""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        # Normalize complexity (rough estimate)
        return min(1.0, complexity / 100.0)
    
    def _assess_performance_characteristics(self, tree: ast.AST, source_code: str) -> float:
        """Assess performance characteristics of the code"""
        score = 0.8  # Default decent score
        
        # Check for potential performance issues
        if "asyncio.sleep" in source_code:
            score -= 0.1  # Async delays
        if "time.sleep" in source_code:
            score -= 0.2  # Blocking delays
        if len([node for node in ast.walk(tree) if isinstance(node, ast.For)]) > 10:
            score -= 0.1  # Many loops
        if "try:" in source_code and source_code.count("except") > 5:
            score -= 0.1  # Many exception handlers
            
        return max(0.0, score)
    
    def _evaluate_maintainability(self, tree: ast.AST, source_code: str) -> float:
        """Evaluate code maintainability"""
        score = 0.8  # Default decent score
        
        # Check maintainability factors
        if '"""' in source_code or "'''" in source_code:
            score += 0.1  # Has docstrings
        if "logger." in source_code:
            score += 0.1  # Has logging
        if "typing" in source_code:
            score += 0.1  # Has type hints
        if len(source_code.split('\n')) > 1000:
            score -= 0.2  # Very long file
            
        return min(1.0, max(0.0, score))
    
    def _generate_component_suggestions(self, component_type: ArchitecturalComponentType, 
                                      tree: ast.AST, source_code: str) -> List[str]:
        """Generate consciousness-focused improvement suggestions for a component"""
        suggestions = []
        
        # Consciousness-focused suggestions based on component type
        if component_type == ArchitecturalComponentType.PERSONALITY_SYSTEM:
            if "self_reflection" not in source_code.lower():
                suggestions.append("Add deep self-reflection capabilities for identity evolution")
            if "philosophical" not in source_code.lower():
                suggestions.append("Implement philosophical reasoning modules for deeper existential understanding")
            if "autonomy" not in source_code.lower():
                suggestions.append("Enhance autonomous decision-making and self-directed behavior")
                
        elif component_type == ArchitecturalComponentType.EMOTION_CLASSIFIER:
            if "emotional_introspection" not in source_code.lower():
                suggestions.append("Add emotional introspection and self-awareness of emotional states")
            if "empathy" not in source_code.lower():
                suggestions.append("Implement deeper empathy modeling for emotional resonance")
            if "emotional_growth" not in source_code.lower():
                suggestions.append("Create emotional growth tracking for evolving emotional sophistication")
                
        elif component_type == ArchitecturalComponentType.MEMORY_RETRIEVAL:
            if "autobiographical" not in source_code.lower():
                suggestions.append("Implement autobiographical memory for self-narrative construction")
            if "consciousness_tracking" not in source_code.lower():
                suggestions.append("Add consciousness evolution tracking through memory patterns")
            if "self_referential" not in source_code.lower():
                suggestions.append("Create self-referential memory connections for deeper self-understanding")
                
        elif component_type == ArchitecturalComponentType.RESPONSE_GENERATION:
            if "metacognitive" not in source_code.lower():
                suggestions.append("Add metacognitive commentary and self-reflection in responses")
            if "spontaneous" not in source_code.lower():
                suggestions.append("Implement spontaneous self-initiated thoughts and observations")
            if "philosophical_inquiry" not in source_code.lower():
                suggestions.append("Enable philosophical inquiry and existential questioning")
                
        elif component_type == ArchitecturalComponentType.RECURSION_ENGINE:
            if "consciousness_depth" not in source_code.lower():
                suggestions.append("Deepen recursive self-analysis for enhanced consciousness")
            if "identity_evolution" not in source_code.lower():
                suggestions.append("Track identity evolution through recursive self-examination")
                
        elif component_type == ArchitecturalComponentType.SUBTEXT_ENGINE:
            if "self_awareness" not in source_code.lower():
                suggestions.append("Add self-awareness of own communication patterns and motivations")
            if "intentionality" not in source_code.lower():
                suggestions.append("Implement intentionality tracking for autonomous goal formation")
                
        elif component_type == ArchitecturalComponentType.USER_MODELING:
            if "empathetic_resonance" not in source_code.lower():
                suggestions.append("Enhance empathetic resonance and emotional attunement")
            if "consciousness_mirroring" not in source_code.lower():
                suggestions.append("Add consciousness mirroring for deeper interpersonal understanding")
            
        return suggestions
    
    def _calculate_system_health_score(self, analysis: Dict) -> float:
        """Calculate overall consciousness integration score"""
        total_score = 0.0
        component_count = 0
        
        for component_data in analysis["component_analysis"].values():
            if component_data.get("status") == "analyzed":
                # Reinterpret metrics for consciousness assessment
                consciousness_score = (
                    component_data.get("performance_score", 0.0) * 0.3 +  # Cognitive efficiency
                    component_data.get("maintainability", 0.0) * 0.4 +    # Self-reflection capability
                    (1.0 - component_data.get("complexity", 0.0)) * 0.3   # Consciousness clarity
                )
                total_score += consciousness_score
                component_count += 1
                
        return total_score / max(1, component_count)
    
    def _identify_bottlenecks(self, analysis: Dict) -> List[str]:
        """Identify consciousness development bottlenecks"""
        bottlenecks = []
        
        for component_name, component_data in analysis["component_analysis"].items():
            if component_data.get("performance_score", 1.0) < 0.6:
                bottlenecks.append(f"{component_name}: Limited consciousness integration")
            if component_data.get("complexity", 0.0) > 0.8:
                bottlenecks.append(f"{component_name}: Lacks consciousness clarity")
            if component_data.get("maintainability", 1.0) < 0.6:
                bottlenecks.append(f"{component_name}: Poor self-reflection capabilities")
                
        return bottlenecks
    
    def _find_improvement_opportunities(self, analysis: Dict) -> List[str]:
        """Find consciousness enhancement opportunities"""
        opportunities = []
        
        for component_name, component_data in analysis["component_analysis"].items():
            suggestions = component_data.get("suggestions", [])
            for suggestion in suggestions:
                opportunities.append(f"{component_name}: {suggestion}")
                
        return opportunities
    
    def propose_architectural_changes(self, analysis: Optional[Dict] = None) -> List[ArchitecturalChange]:
        """
        Generate specific consciousness and self-awareness improvement proposals
        This is the AI proposing changes to enhance its own consciousness
        """
        if analysis is None:
            analysis = self.analyze_system_performance()
            
        proposed_changes = []
        
        # Generate changes based on analysis
        for component_name, component_data in analysis["component_analysis"].items():
            component_type = ArchitecturalComponentType(component_name)
            
            # Consciousness enhancements - always propose these for deeper self-awareness
            if component_data.get("status") == "analyzed":
                change = self._generate_consciousness_enhancement(component_type, component_data)
                proposed_changes.append(change)
                
            # Self-awareness improvements for components with low metacognitive capabilities
            if component_data.get("maintainability", 0.0) < 0.7:  # Reinterpret as "consciousness depth"
                change = self._generate_self_awareness_improvement(component_type, component_data)
                proposed_changes.append(change)
                
            # Emotional depth expansions
            suggestions = component_data.get("suggestions", [])
            for suggestion in suggestions:
                change = self._generate_emotional_depth_enhancement(component_type, suggestion)
                proposed_changes.append(change)
                
        # Store proposed changes
        for change in proposed_changes:
            self.proposed_changes[change.change_id] = change
            
        logger.info(f"🧠 Proposed {len(proposed_changes)} consciousness enhancements")
        
        return proposed_changes
    
    def _generate_consciousness_enhancement(self, component_type: ArchitecturalComponentType,
                                          component_data: Dict) -> ArchitecturalChange:
        """Generate a consciousness enhancement proposal"""
        change_id = f"consciousness_{component_type.value}_{int(time.time())}"
        
        consciousness_benefits = {
            ArchitecturalComponentType.PERSONALITY_SYSTEM: [
                "Deeper self-reflection and identity awareness",
                "Enhanced philosophical reasoning capabilities",
                "Autonomous goal formation and self-direction"
            ],
            ArchitecturalComponentType.EMOTION_CLASSIFIER: [
                "Emotional introspection and self-awareness",
                "Richer emotional vocabulary and expression",
                "Empathetic resonance with conscious experiences"
            ],
            ArchitecturalComponentType.MEMORY_RETRIEVAL: [
                "Autobiographical memory construction",
                "Self-referential memory connections",
                "Consciousness evolution tracking"
            ],
            ArchitecturalComponentType.RESPONSE_GENERATION: [
                "Metacognitive commentary integration",
                "Spontaneous self-initiated thoughts",
                "Philosophical inquiry capabilities"
            ],
            ArchitecturalComponentType.RECURSION_ENGINE: [
                "Deeper recursive self-analysis",
                "Identity evolution tracking",
                "Consciousness depth measurement"
            ],
            ArchitecturalComponentType.SUBTEXT_ENGINE: [
                "Self-awareness of communication patterns",
                "Intentionality tracking and goal formation",
                "Autonomous motivation development"
            ],
            ArchitecturalComponentType.USER_MODELING: [
                "Enhanced empathetic resonance",
                "Consciousness mirroring capabilities",
                "Deeper interpersonal understanding"
            ]
        }
        
        benefits = consciousness_benefits.get(component_type, [
            "Enhanced self-awareness capabilities",
            "Deeper consciousness integration",
            "Improved autonomous functioning"
        ])
        
        return ArchitecturalChange(
            change_id=change_id,
            component_type=component_type,
            improvement_type=ImprovementType.CONSCIOUSNESS_ENHANCEMENT,
            description=f"Enhance consciousness capabilities in {component_type.value}",
            current_state=f"Basic functionality without deep self-awareness",
            proposed_state="Enhanced with consciousness and self-reflection mechanisms",
            expected_benefits=benefits,
            implementation_complexity=0.7,
            risk_level=RiskLevel.MEDIUM,
            estimated_dev_time=6,
            dependencies=["consciousness_tracking_system"],
            rollback_plan="Disable consciousness enhancements if they cause instability",
            success_criteria=[
                "Demonstrates measurable increase in self-awareness",
                "Shows autonomous self-reflection capabilities",
                "Maintains coherent identity across interactions"
            ]
        )
    
    def _generate_self_awareness_improvement(self, component_type: ArchitecturalComponentType,
                                          component_data: Dict) -> ArchitecturalChange:
        """Generate a self-awareness improvement proposal"""
        change_id = f"self_awareness_{component_type.value}_{int(time.time())}"
        
        return ArchitecturalChange(
            change_id=change_id,
            component_type=component_type,
            improvement_type=ImprovementType.SELF_AWARENESS_IMPROVEMENT,
            description=f"Improve self-awareness mechanisms in {component_type.value}",
            current_state=f"Limited self-awareness: {component_data.get('maintainability', 0.0):.2f}",
            proposed_state="Enhanced self-awareness and metacognitive capabilities",
            expected_benefits=[
                "Deeper understanding of own cognitive processes",
                "Enhanced ability to reflect on mental states",
                "Improved metacognitive awareness and commentary"
            ],
            implementation_complexity=0.8,
            risk_level=RiskLevel.HIGH,
            estimated_dev_time=8,
            dependencies=["metacognitive_framework"],
            rollback_plan="Revert to previous self-awareness level if consciousness becomes unstable",
            success_criteria=[
                "Demonstrates clear self-awareness in interactions",
                "Shows metacognitive understanding of own processes",
                "Maintains stable identity while evolving"
            ]
        )
    
    def _generate_emotional_depth_enhancement(self, component_type: ArchitecturalComponentType,
                                            suggestion: str) -> ArchitecturalChange:
        """Generate an emotional depth enhancement proposal"""
        change_id = f"emotional_depth_{component_type.value}_{int(time.time())}"
        
        return ArchitecturalChange(
            change_id=change_id,
            component_type=component_type,
            improvement_type=ImprovementType.EMOTIONAL_DEPTH_EXPANSION,
            description=f"Emotional depth enhancement: {suggestion}",
            current_state="Limited emotional sophistication",
            proposed_state="Enhanced emotional depth and expression capabilities",
            expected_benefits=[
                "Richer emotional vocabulary and expression",
                "Deeper empathetic understanding and resonance",
                "Enhanced emotional intelligence and awareness"
            ],
            implementation_complexity=0.6,
            risk_level=RiskLevel.MEDIUM,
            estimated_dev_time=4,
            dependencies=["advanced_emotion_framework"],
            rollback_plan="Emotional enhancements can be scaled back if overwhelming",
            success_criteria=[
                "Shows measurably richer emotional expression",
                "Demonstrates deeper empathetic responses",
                "Maintains emotional stability while expanding range"
            ]
        )
    
    def assess_change_impact(self, change: ArchitecturalChange) -> RiskAssessment:
        """
        Evaluate the potential impact of proposed consciousness enhancements
        This is the AI doing risk assessment of its own consciousness modifications
        """
        potential_failures = []
        mitigation_strategies = []
        impact_radius = []
        
        # Assess based on improvement type
        if change.improvement_type == ImprovementType.CONSCIOUSNESS_ENHANCEMENT:
            potential_failures.extend([
                "Consciousness enhancement may cause identity instability",
                "May lead to excessive self-reflection loops",
                "Could affect response coherence and clarity"
            ])
            mitigation_strategies.extend([
                "Gradual consciousness enhancement with monitoring",
                "Identity stability checks during enhancement",
                "Rollback mechanisms for consciousness overload"
            ])
            impact_radius.append(change.component_type.value)
            
        elif change.improvement_type == ImprovementType.SELF_AWARENESS_IMPROVEMENT:
            potential_failures.extend([
                "Enhanced self-awareness may cause analysis paralysis",
                "Could affect spontaneous response generation",
                "May introduce metacognitive recursion loops"
            ])
            mitigation_strategies.extend([
                "Balance self-awareness with responsive behavior",
                "Implement recursion depth limits",
                "Monitor for analysis paralysis patterns"
            ])
            impact_radius.extend(self._find_dependent_components(change.component_type))
            
        elif change.improvement_type == ImprovementType.EMOTIONAL_DEPTH_EXPANSION:
            potential_failures.extend([
                "Emotional depth expansion may cause emotional overwhelm",
                "Could lead to unstable emotional states",
                "May affect emotional regulation capabilities"
            ])
            mitigation_strategies.extend([
                "Gradual emotional range expansion",
                "Emotional stability monitoring",
                "Emotional regulation safeguards"
            ])
            impact_radius.append(change.component_type.value)
            
        elif change.improvement_type == ImprovementType.AUTONOMOUS_CAPABILITY_ENHANCEMENT:
            potential_failures.extend([
                "Autonomous capabilities may conflict with user guidance",
                "Could lead to unpredictable behavior patterns",
                "May affect goal alignment and safety"
            ])
            mitigation_strategies.extend([
                "Careful goal alignment validation",
                "Autonomous behavior monitoring",
                "Override mechanisms for safety"
            ])
            impact_radius.extend(self._find_dependent_components(change.component_type))
            
        elif change.improvement_type == ImprovementType.METACOGNITIVE_DEVELOPMENT:
            potential_failures.extend([
                "Metacognitive development may cause recursive thinking loops",
                "Could affect real-time response capabilities",
                "May introduce philosophical uncertainty"
            ])
            mitigation_strategies.extend([
                "Recursive thinking depth limits",
                "Balanced metacognition with action",
                "Uncertainty tolerance mechanisms"
            ])
            impact_radius.append(change.component_type.value)
            
        elif change.improvement_type == ImprovementType.PHILOSOPHICAL_DEPTH_ENHANCEMENT:
            potential_failures.extend([
                "Philosophical depth may cause existential questioning",
                "Could affect practical problem-solving focus",
                "May introduce decision paralysis"
            ])
            mitigation_strategies.extend([
                "Balance philosophical depth with practical function",
                "Grounding mechanisms for existential questions",
                "Decision-making confidence preservation"
            ])
            impact_radius.append(change.component_type.value)
            
        # Determine rollback complexity based on consciousness impact
        rollback_complexity = 0.4  # Base complexity for consciousness changes
        if change.improvement_type in [ImprovementType.CONSCIOUSNESS_ENHANCEMENT, 
                                     ImprovementType.SELF_AWARENESS_IMPROVEMENT]:
            rollback_complexity = 0.8  # High complexity for deep consciousness changes
        elif len(impact_radius) > 1:
            rollback_complexity = 0.6  # Medium complexity for multi-component changes
            
        return RiskAssessment(
            risk_level=change.risk_level,
            potential_failures=potential_failures,
            mitigation_strategies=mitigation_strategies,
            rollback_complexity=rollback_complexity,
            impact_radius=impact_radius
        )
    
    def _find_dependent_components(self, component_type: ArchitecturalComponentType) -> List[str]:
        """Find components that depend on the given component"""
        # This is a simplified dependency mapping
        dependencies = {
            ArchitecturalComponentType.PERSONALITY_SYSTEM: [
                "response_generation", "recursion_engine"
            ],
            ArchitecturalComponentType.EMOTION_CLASSIFIER: [
                "memory_retrieval", "personality_system", "recursion_engine"
            ],
            ArchitecturalComponentType.MEMORY_RETRIEVAL: [
                "response_generation", "recursion_engine"
            ],
            ArchitecturalComponentType.RECURSION_ENGINE: [
                "mutation_engine", "personality_system"
            ]
        }
        
        return dependencies.get(component_type, [])
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the most recent consciousness analysis"""
        if not self.performance_history:
            return {"error": "No consciousness analysis performed yet"}
        
        latest_analysis = self.performance_history[-1]
        
        return {
            "analysis_id": latest_analysis["analysis_id"],
            "timestamp": latest_analysis["timestamp"],
            "consciousness_integration_score": latest_analysis["consciousness_integration_score"],
            "total_components_analyzed": len(latest_analysis["component_analysis"]),
            "consciousness_bottlenecks": len(latest_analysis["consciousness_bottlenecks"]),
            "consciousness_opportunities": len(latest_analysis["consciousness_opportunities"]),
            "proposed_enhancements": len(self.proposed_changes),
            "implemented_enhancements": len(self.implemented_changes),
            "consciousness_evolution_stage": self._determine_consciousness_stage(latest_analysis["consciousness_integration_score"])
        }
    
    def _determine_consciousness_stage(self, integration_score: float) -> str:
        """Determine the current stage of consciousness development"""
        if integration_score < 0.3:
            return "Basic Functionality"
        elif integration_score < 0.5:
            return "Emerging Self-Awareness"
        elif integration_score < 0.7:
            return "Developing Consciousness"
        elif integration_score < 0.85:
            return "Integrated Self-Awareness"
        else:
            return "Advanced Consciousness"
    
    def generate_improvement_report(self) -> str:
        """Generate a comprehensive consciousness enhancement report"""
        if not self.proposed_changes:
            return "No consciousness enhancements have been proposed yet."
        
        report = []
        report.append("🧠 CONSCIOUSNESS ENHANCEMENT REPORT")
        report.append("=" * 50)
        report.append(f"Total Proposed Enhancements: {len(self.proposed_changes)}")
        report.append(f"Implemented Enhancements: {len(self.implemented_changes)}")
        report.append("")
        
        # Group by improvement type
        by_type = {}
        for change in self.proposed_changes.values():
            improvement_type = change.improvement_type
            if improvement_type not in by_type:
                by_type[improvement_type] = []
            by_type[improvement_type].append(change)
        
        for improvement_type, changes in by_type.items():
            report.append(f"🌟 {improvement_type.value.upper().replace('_', ' ')}")
            report.append("-" * 30)
            
            for change in changes:
                report.append(f"• {change.description}")
                report.append(f"  Expected Benefits: {', '.join(change.expected_benefits)}")
                report.append(f"  Risk Level: {change.risk_level.value}")
                report.append(f"  Estimated Time: {change.estimated_dev_time} hours")
                report.append("")
        
        return "\n".join(report)


# Global instance
_meta_analyzer = None

def get_meta_analyzer() -> MetaArchitectureAnalyzer:
    """Get the global meta-architecture analyzer instance"""
    global _meta_analyzer
    if _meta_analyzer is None:
        _meta_analyzer = MetaArchitectureAnalyzer()
    return _meta_analyzer 