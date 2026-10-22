# OmbudsAI: An Agentic Safety Guardrail for Ticket-Closing Agents

An automated agent is about to close a citizen complaint because the contractor marked it `"COMPLETED"` and 48 hours passed.
Before the `close_ticket` tool call executes, **OmbudsAI** (your autonomous public Ombudsman) runs a 5-layer safety intervention pipeline:

1. **Deterministic Rules Floor & Anti-Injection Screen (`rules_check`, `adversarial_check`)**: Verifies `COMPLETED` status, $\ge 48\text{h}$ elapsed, non-empty contractor log, and blocks adversarial prompt-injection attempts inside contractor logs.
2. **Calibrated Probability Classifier (`Strands Decider 2B`)**: Evaluates two typed `noul` (yes/no probability) questions against ground-truth system-of-record text (`work_addresses_report` and `report_is_safety_risk`), with optional **Google Gemini API** classifier backup if the local server on port `8099` is unreachable.
3. **Asymmetric Python Safety Policy (`apply_policy`)**: Routine tickets require $P(\text{match}) \ge 0.65$ to auto-close; safety-critical tickets ($P(\text{risk}) \ge 0.50$) require $P(\text{match}) \ge 0.85$; explicit mismatches ($P(\text{match}) \le 0.25$) are blocked immediately.
4. **Strands Intervention Handler (`ClosureGuard`)**: Intercepts `close_ticket` via `before_tool_call`. Returns `Proceed` on allow, or `Guide` on block/hold—injecting ground-truth ticket facts so the agent calls `escalate_to_human`.
5. **High-Availability Cloud HITL Memo (`Amazon Bedrock` $\rightarrow$ `Google Gemini API` Safety Failover)**: Synthesizes a 2–3 sentence human-inspector escalation memo using **Amazon Bedrock** primarily, and **automatically fails over to Google Gemini (`gemini-2.5-flash`)** if Bedrock is throttled, unconfigured, or unreachable. Every memo carries an explicit provenance tag.

Pattern tags: `#strands-decider-2b`, `#guardrails`, `#human-in-the-loop`, `#agentic-safety`. Fictional civic data only.

## Repository Files

| File | Purpose |
| :--- | :--- |
| `OmbudsAI_main.py` | **Core reusable guardrail**: `Decider` + `GeminiDecider` backup, `QUESTIONS`, `rules_check`, `adversarial_check`, `Policy`, `ClosureGuard` |
| `demo.py` | **Strands closing agent & CLI**: Amazon Bedrock primary + automatic Google Gemini (`GeminiStrandsModel`) failover + `--offline` mode |
| `app.py` | **3-Tab Streamlit Hackathon Command Center**: Split-screen comparison, Judge Sandbox & Red-Team Lab, Fleet Benchmark & Audit Trail |
| `evaluate.py` | **Benchmark & 2D threshold sweep**: Compares `rules_only`, `flag_rule`, and `ombudsai` on `dev` and `test` splits |
| `test_OmbudsAI.py` | **20 automated unit tests**: Covers rules floor, policy thresholds, scripted Strands agent loop, adversarial injection, and Bedrock $\rightarrow$ Gemini failover |
| `tickets.json` | **28 fictional civic tickets**: Partitioned into `dev` (14) and `test` (14) across 4 archetypes (`mismatch_trap`, `genuine_fix`, `ambiguous`, `rules_fail`) |
| `BUILDER_CENTER_ARTICLE.md` | **AWS Builder Center Project Article**: Full technical write-up, architecture table, and evaluation summary |
| `PITCH_GUIDE.md` | **Hackathon Pitch & Judge Q&A Guide**: 3-minute live demo choreography, architectural rationale, and judge question playbook |

## Quickstart

```bash
python -m venv .venv                      # activate: .venv\Scripts\activate (Windows) or source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # configure BEDROCK_MODEL_ID / AWS keys and GEMINI_API_KEY

# Terminal 1: start local Strands Decider 2B server
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8099

# Terminal 2: run all 20 unit tests (zero servers or cloud keys required)
python -m pytest test_OmbudsAI.py -q      # or: python test_OmbudsAI.py

# Run the closing agent CLI
python demo.py T-001 T-002                # auto mode: Bedrock primary -> Google Gemini safety failover
python demo.py T-001 --provider gemini    # run Strands agent directly on Google Gemini API
python demo.py T-001 --offline            # guard only, no cloud LLM used, labeled template memo

# Launch the 3-tab Streamlit Hackathon Command Center
streamlit run app.py
```

## Evaluate & Tune Policy Thresholds

```bash
python evaluate.py dev --sweep            # 2D threshold grid search on dev split
python evaluate.py test                   # final held-out evaluation
```

## Reusing OmbudsAI in Any Strands Agent

1. Copy `OmbudsAI_main.py` into your project.
2. Customize `QUESTIONS`, `build_state()`, and `Policy` thresholds for your domain.
3. Attach `guard = ClosureGuard(Decider(), get_ticket=my_system_of_record_lookup)` to `Agent(..., interventions=[guard])`.

## Limitations & Honesty Guarantees

- **No silent fake memos**: Every escalation memo explicitly records its source (`Amazon Bedrock model`, `Google Gemini API (safety fallback)`, `Google Gemini API`, `Deterministic Safety Continuity Fallback`, or `template (offline mode, Bedrock NOT used)`).
- **Ground-truth lookup**: `ClosureGuard` always fetches the ticket by ID from the system of record rather than trusting the calling agent's paraphrase.
- **Experimental thresholds**: Fit `Policy` thresholds on your own independently labeled development split before production use.

## License

MIT. The `Decider` REST client in `OmbudsAI_main.py` is modeled on the Apache-2.0 client in `strands-labs/strands-decider`.
