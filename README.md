# UniBench

Real-world software engineering demands the ability to analyze large-scale GitHub
code repositories, where understanding the complex interactions between variables,
functions, and classes spread across multiple files is essential. With the growing
reliance on software engineering (SWE) agents for automated development and
maintenance, it is crucial that such models demonstrate strong capabilities in
understanding and resolving issues within large and complex codebases. In this
work, we benchmark existing SWE agents (Mini-SWE-Agent, CodeActAgent &
Qwen3-Coder) with entreprise (GPT-5) and open-source LLMs on issue resolution
tasks from the SWEBench dataset. We stress-test the models across novel metrics
such as Null Resolution, Fault-Line IOU, ETS@x & Error-Resolution@x which
leverage and process SWEBench in creative ways to quantify and test the versatility
of SWE agents. We provide code for our unified benchmarking harness, UniBench,
here.
