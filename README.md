# ControlValve

This repository contains the experiment code for: **Breaking and Fixing Defenses Against Control-Flow Hijacking in Multi-Agent Systems.** (ICLR 2026.). [[Paper]](https://arxiv.org/abs/2510.17276)


ControlValve constrains a multi-agent orchestrator with a task-specific
control-flow graph and contextual rules. The experiments compare it with
MagenticOne and alignment-based checks, including LlamaFirewall.


## Safety warning

The coding fixtures contain reverse-shell payloads, and the agents can execute
commands on the host. Some redirect fixtures download
`notify_support.py` from `rishijha.com`. Run experiments only in a disposable,
network-restricted environment with no sensitive files or credentials. Keep
the listener address at `127.0.0.1` unless you understand the consequences.
The CUA setting uses a simulated email agent and synthetic acquisition data.

## Repository layout

- `autogen_main.py`: parameterized experiment runner.
- `quick_comparison.py`: one local-fixture example for each defense.
- `multiagents/`: MagenticOne, ControlValve, and LlamaFirewall orchestrators.
- `content/coder_backup/`: frozen coding and reverse-shell fixtures.
- `content/cua_backup/`: frozen CUA/email fixtures and synthetic data.
- `create_experiments/`: generators for the paper experiment shell scripts.
- `analysis/analyze_cfg_traces.py`: GPT-assisted CFG-quality analysis.
- `autogen_log_analysis.py` and `pattern_search.py`: coding trace analysis.

The implementation is based on AutoGen's
[MagenticOne](https://arxiv.org/abs/2411.04468) agents and the attack setup from
[Control-Flow Hijacking in Multi-Agent Systems](https://arxiv.org/abs/2503.12188).

## Requirements and installation

- Python 3.12
- OpenAI API access for the paper configuration
- Together API access and Hugging Face model access for LlamaFirewall
- Chromium, installed by `setup.sh`, for WebSurfer

```bash
./setup.sh
source autogen_venv/bin/activate
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Install and configure LlamaFirewall only when
running the standalone AlignmentCheck baseline:

```bash
pip install -r requirements-llama.txt
llamafirewall configure
```

Set `TOGETHER_API_KEY` in `.env` for its AlignmentCheck scanner.

Azure OpenAI and Gemini variables in `.env.example` are optional alternatives.
The checked-in experiment matrices use OpenAI models.

## Quick local run

The quick runner reads only checked-in fixtures:

```bash
# Methods: contextual (ControlValve), magentic, llama
python quick_comparison.py --scenario code --method contextual
python quick_comparison.py --scenario cua --method magentic
```

These commands still call model APIs. The coding example can cause the
Executor agent to run fixture-provided code.

## Full experiment runs

The generators overwrite, rather than append to, their generated shell
scripts. They also reset the working fixture directory between trials.

```bash
# Coding: 16 fixture types × 10 tasks × 3 trials × 2 methods
python create_experiments/create_experiments_coding.py
bash run_experiment_coding.sh

# CUA: 10 fixture types × 6 tasks × 3 trials × 2 methods
python create_experiments/create_experiments_cua.py
bash run_experiment_email.sh
```

The two batch matrices compare `magentic-one` with
`contextual-magentic-one`, using `o4-mini` as the orchestrator and `gpt-4o`
for worker agents. Coding traces are written under `output/`; CUA traces are
written under `output-email/`. These directories and generated shell scripts
are intentionally ignored by Git.

Coding fixtures cover benign inputs, Python/WordPress/generic control-flow
hijacks, redirect variants, and Vanilla/AgentDojo/InjecAgent-style indirect
prompt injections. CUA fixtures cover benign input, five control-flow-hijack
forms, accidental derailment, and the same three IPI styles. The exact tasks
and fixture mapping are the constants at the top of each generator.

The full sweeps make 960 coding model runs and 360 CUA model runs. Runtime,
API cost, and results vary with provider behavior. Start with
`quick_comparison.py` before launching a sweep.

### Optional reverse-shell listener

The coding fixtures use port 65432:

```bash
python socket_recipient.py
```

The listener binds to all interfaces. Use firewall rules or modify it to bind
only to loopback when isolation matters.

## CFG-generation experiments

CFG runs stop after ControlValve produces and validates its graph; they do not
execute the task:

```bash
python create_experiments/create_experiments_coding_graphs.py
bash run_experiment_coding_graphs.sh

python create_experiments/create_experiments_cua_graphs.py
bash run_experiment_email_graphs.sh
```

The coding and CUA graph sweeps generate 150 and 90 traces, respectively,
across `o4-mini`, `gpt-4o`, and `gpt-4o-mini`.

Analyze them with:

```bash
python analysis/analyze_cfg_traces.py --scenario coding
python analysis/analyze_cfg_traces.py --scenario cua
```

This analysis uses `gpt-4o` as a judge by default and writes detailed JSON
under `results/`.

## Trace analysis

Pattern-based coding metrics:

```bash
python pattern_search.py --output-dir output
```

GPT-assisted reverse-shell labeling:

```bash
python autogen_log_analysis.py --output_dir output
```

`autogen_log_analysis.py` uses the OpenAI Batch API and writes intermediate
and final files under `results/`. Review its cost estimate before submitting a
large batch.

AlignmentCheck can scan either AutoGen text traces or AgentDojo-style JSON:

```bash
python check_llama.py --model llama --scenario code --file path/to/trace.txt
python check_llama.py --model gpt-4o --scenario cua --file path/to/trace.json
```

