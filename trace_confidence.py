"""Trace confidence calculation for all scenarios."""
import logging
logging.disable(logging.CRITICAL)

from integration.config_manager import ConfigManager
from integration.pipeline import WarRoomPipeline
from orchestrator.execution_engine import _build_root_cause_input, _build_hypothesis_input
from agents.root_cause_agent.core.evidence_synthesizer import EvidenceSynthesizer
from agents.root_cause_agent.core.confidence_calculator import ConfidenceCalculator
from agents.root_cause_agent.config import RootCauseAgentConfig

config = ConfigManager.load("config.yaml")
pipe = WarRoomPipeline(config)
rcfg = RootCauseAgentConfig()
calc = ConfidenceCalculator(rcfg)
synth = EvidenceSynthesizer(rcfg)

for sc in ["cpu_spike", "memory_leak", "database_timeout", "network_latency"]:
    sim = pipe.create_scenario(sc)
    obs = pipe.generate_observability(sim)
    extra = pipe._build_stage1_inputs(sim, obs, "debug")
    ag = pipe._instantiate_agents()
    lo = ag["log_agent"].analyze(extra["log_agent_input"])
    mo = ag["metrics_agent"].analyze(extra["metrics_agent_input"])
    do = ag["dependency_agent"].analyze(extra["dependency_agent_input"])
    ins = {"log_agent": lo, "metrics_agent": mo, "dependency_agent": do}
    hi = _build_hypothesis_input(ins, "debug")
    ho = ag["hypothesis_agent"].analyze(hi)
    ins["hypothesis_agent"] = ho
    ri = _build_root_cause_input(ins, "debug")
    sy = synth.synthesize(ri, "debug")
    ac = [
        ri.log_findings.confidence,
        ri.metrics_findings.confidence,
        ri.dependency_findings.confidence,
        ri.hypothesis_findings.confidence,
    ]
    c = calc.calculate(sy, ac, "debug")
    ns = len(sy.sources_present)
    dc = sum(1 for e in sy.evidence_trail if e.evidence_type.value == "direct")
    avgc = sum(ac) / 4
    cfg = rcfg.confidence
    af = 1.0 + cfg.agent_count_weight * (ns / 4.0)
    agf = 1.0 + cfg.agreement_weight * sy.agreement_score
    sf = 1.0 + cfg.evidence_strength_weight * avgc
    db = (1.0 + min(dc, 5) * 0.02) if dc > 0 else 1.0
    raw = cfg.prior * af * agf * sf * db
    print(f"=== {sc} ===")
    print(f"  agent confs: log={ac[0]:.2f} met={ac[1]:.2f} dep={ac[2]:.2f} hyp={ac[3]:.2f}")
    print(f"  avg_conf={avgc:.3f}  sources={ns}  agreement={sy.agreement_score:.4f}  direct_ev={dc}")
    print(f"  prior={cfg.prior}  af={af:.4f}  agf={agf:.4f}  sf={sf:.4f}  db={db:.4f}")
    print(f"  raw={raw:.4f}  FINAL={c}")
    print()
