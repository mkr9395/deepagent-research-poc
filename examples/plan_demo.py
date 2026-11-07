"""Planning-only demo: does the task decomposition change with the request?

Runs the REAL orchestrator prompt and plan validation with the real model, but with no
researcher and no Tavily, so it costs a couple of LLM calls and no searches.

    python -m examples.plan_demo
"""
from deepagents import create_deep_agent

from fin_agent.agents.orchestrator import ORCHESTRATOR_PROMPT, ResearchPlanner, _model_name

REQUESTS = [
    "Compare TESLA and Apple based on revenue growth and profitability.",
    "Compare Amazon and Walmart based on revenue performance, profitability, "
    "major regulatory risks and business outlook.",
]
DRY_RUN = "\n\n(Planning dry run: call submit_research_plan once with an accepted plan, then stop.)"


def main() -> None:
    for request in REQUESTS:
        planner = ResearchPlanner()
        agent = create_deep_agent(
            model=_model_name(), tools=[planner.submit_research_plan], system_prompt=ORCHESTRATOR_PROMPT
        )
        agent.invoke({"messages": [{"role": "user", "content": request + DRY_RUN}]})
        print(f"\nREQUEST: {request}")
        if not planner.plan:
            print("  (no plan accepted)")
            continue
        print(f"  requested dimensions: {planner.plan['dimensions']}")
        print(f"  {len(planner.plan['tasks'])} research task(s):")
        for t in planner.plan["tasks"]:
            print(f"   - {t['dimensions']}: {t['question']}")
        print(f"  skipped: {planner.plan['skipped'] or 'none'}")


if __name__ == "__main__":
    main()
