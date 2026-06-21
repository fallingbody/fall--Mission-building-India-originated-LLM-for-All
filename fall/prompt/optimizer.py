"""
Automatic prompt optimization for FALL.
Uses evolutionary search to find optimal prompts for specific tasks.
"""
import random
import copy
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json

@dataclass
class PromptCandidate:
    text: str
    score: float = 0.0
    generation_count: int = 0
    success_count: int = 0

class PromptOptimizer:
    def __init__(
        self,
        model_api,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        elite_count: int = 4,
        max_generations: int = 50,
    ):
        self.model = model_api
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.max_generations = max_generations
        self.population: List[PromptCandidate] = []
    
    def optimize(
        self,
        task_description: str,
        evaluation_fn: callable,
        initial_prompts: Optional[List[str]] = None,
    ) -> PromptCandidate:
        """Optimize prompts for a given task."""
        # Initialize population
        if initial_prompts:
            self.population = [
                PromptCandidate(text=p) for p in initial_prompts[:self.population_size]
            ]
        
        # Fill remaining with variations
        while len(self.population) < self.population_size:
            base = random.choice(self.population) if self.population else PromptCandidate(
                text=f"Complete the following task: {{task}}\n\nTask: {task_description}"
            )
            mutated = self._mutate(base)
            self.population.append(mutated)
        
        # Evolutionary loop
        for gen in range(self.max_generations):
            # Evaluate population
            for candidate in self.population:
                if candidate.generation_count < 10:  # Limit evaluations
                    score = evaluation_fn(candidate.text)
                    candidate.score = (candidate.score * candidate.generation_count + score) / (candidate.generation_count + 1)
                    candidate.generation_count += 1
            
            # Sort by score
            self.population.sort(key=lambda c: c.score, reverse=True)
            
            # Keep elites
            new_population = self.population[:self.elite_count]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = random.choice(self.population[:self.population_size // 2])
                parent2 = random.choice(self.population[:self.population_size // 2])
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                child.score = 0.0
                child.generation_count = 0
                new_population.append(child)
            
            self.population = new_population
        
        return self.population[0]
    
    def _mutate(self, candidate: PromptCandidate) -> PromptCandidate:
        """Apply random mutations to a prompt."""
        mutations = [
            self._add_instruction,
            self._add_example,
            self._rephrase,
            self._change_tone,
            self._add_constraints,
        ]
        mutator = random.choice(mutations)
        return mutator(candidate)
    
    def _add_instruction(self, candidate: PromptCandidate) -> PromptCandidate:
        new = copy.deepcopy(candidate)
        instructions = [
            "Think step by step.",
            "Be thorough and precise.",
            "Consider edge cases.",
            "Provide code examples.",
            "Explain your reasoning.",
        ]
        new.text = candidate.text + "\n" + random.choice(instructions)
        return new
    
    def _add_example(self, candidate: PromptCandidate) -> PromptCandidate:
        new = copy.deepcopy(candidate)
        new.text = f"Here is an example of a good response:\n[EXAMPLE]\n\nNow, using the same format:\n{candidate.text}"
        return new
    
    def _rephrase(self, candidate: PromptCandidate) -> PromptCandidate:
        # Swap words with synonyms (simplified)
        new = copy.deepcopy(candidate)
        synonyms = {
            "write": "implement",
            "create": "build",
            "find": "discover",
            "explain": "describe",
            "fix": "repair",
        }
        for old, new_word in synonyms.items():
            if random.random() < 0.3:
                new.text = new.text.replace(old, new_word)
        return new
    
    def _change_tone(self, candidate: PromptCandidate) -> PromptCandidate:
        new = copy.deepcopy(candidate)
        tones = ["professional", "casual", "academic", "technical"]
        tone = random.choice(tones)
        new.text = f"[TONE: {tone}]\n{candidate.text}"
        return new
    
    def _add_constraints(self, candidate: PromptCandidate) -> PromptCandidate:
        new = copy.deepcopy(candidate)
        constraints = [
            "Use only standard library.",
            "Optimize for speed.",
            "Include error handling.",
            "Write unit tests.",
            "Follow PEP 8.",
        ]
        new.text = candidate.text + "\nConstraints:\n- " + random.choice(constraints)
        return new
    
    def _crossover(
        self,
        parent1: PromptCandidate,
        parent2: PromptCandidate,
    ) -> PromptCandidate:
        """Combine two prompts."""
        parts1 = parent1.text.split("\n")
        parts2 = parent2.text.split("\n")
        
        min_len = min(len(parts1), len(parts2))
        if min_len == 0:
            return copy.deepcopy(random.choice([parent1, parent2]))
        
        split_point = random.randint(1, min_len)
        
        new_parts = parts1[:split_point] + parts2[split_point:]
        new = PromptCandidate(text="\n".join(new_parts))
        return new