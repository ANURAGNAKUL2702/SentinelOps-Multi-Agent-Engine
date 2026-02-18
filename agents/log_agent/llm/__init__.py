"""
LLM integration layer for the log agent.

Provides circuit-breaker-protected LLM classification with
automatic fallback to the deterministic rule engine.

Modules:
    classifier  — LLM wrapper with retry, caching, circuit breaker
    synthesizer — Root cause synthesis and remediation
"""
