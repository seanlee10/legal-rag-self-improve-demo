"""Create four Arize online evaluators for document summarization quality.

Run once to set up evaluators:
    python scripts/create_arize_evaluators.py

Requires env vars: ARIZE_API_KEY, ARIZE_SPACE_ID, ARIZE_PROJECT_NAME
"""

import os

# Evaluator definitions: name -> evaluation template
EVALUATORS = {
    "accuracy": (
        "You are evaluating a summary of a regulatory/financial document.\n\n"
        "**Document:**\n{input}\n\n"
        "**Summary:**\n{output}\n\n"
        "Rate the ACCURACY of the summary on a scale of 1-5.\n"
        "- 5: Every fact, figure, and reference in the summary is faithful to the source document.\n"
        "- 4: Minor inaccuracies that do not change the meaning.\n"
        "- 3: Some inaccuracies that could mislead a reader.\n"
        "- 2: Multiple factual errors or hallucinated content.\n"
        "- 1: The summary contains fabricated information not in the source.\n\n"
        'Respond with a JSON object: {"score": <1-5>, "explanation": "<1-2 sentences>"}'
    ),
    "conciseness": (
        "You are evaluating a summary of a regulatory/financial document.\n\n"
        "**Document:**\n{input}\n\n"
        "**Summary:**\n{output}\n\n"
        "Rate the CONCISENESS of the summary on a scale of 1-5.\n"
        "- 5: Tight and efficient — no redundancy, no filler, every sentence adds value.\n"
        "- 4: Mostly concise with minor repetition.\n"
        "- 3: Some unnecessary verbosity or repeated points.\n"
        "- 2: Significantly verbose with substantial redundancy.\n"
        "- 1: The summary is as long as or longer than necessary, defeating the purpose.\n\n"
        'Respond with a JSON object: {"score": <1-5>, "explanation": "<1-2 sentences>"}'
    ),
    "preciseness": (
        "You are evaluating a summary of a regulatory/financial document.\n\n"
        "**Document:**\n{input}\n\n"
        "**Summary:**\n{output}\n\n"
        "Rate the PRECISENESS of the summary on a scale of 1-5.\n"
        "- 5: Preserves all specific details — section numbers, dates, thresholds, named entities.\n"
        "- 4: Most specific details preserved, one or two generalized.\n"
        "- 3: Some specific details replaced with vague language.\n"
        "- 2: Many specific references lost or generalized.\n"
        "- 1: Almost entirely vague — reads like a generic summary.\n\n"
        'Respond with a JSON object: {"score": <1-5>, "explanation": "<1-2 sentences>"}'
    ),
    "completeness": (
        "You are evaluating a summary of a regulatory/financial document.\n\n"
        "**Document:**\n{input}\n\n"
        "**Summary:**\n{output}\n\n"
        "Rate the COMPLETENESS of the summary on a scale of 1-5.\n"
        "- 5: All major provisions, requirements, and obligations are covered.\n"
        "- 4: One minor provision missing.\n"
        "- 3: A few notable provisions or requirements missing.\n"
        "- 2: Major sections or requirements omitted.\n"
        "- 1: The summary covers only a fraction of the document.\n\n"
        'Respond with a JSON object: {"score": <1-5>, "explanation": "<1-2 sentences>"}'
    ),
}


def main() -> None:
    """Print evaluator templates for Arize configuration."""
    space_id = os.environ.get("ARIZE_SPACE_ID", "<not set>")
    api_key = os.environ.get("ARIZE_API_KEY", "<not set>")
    project_name = os.environ.get("ARIZE_PROJECT_NAME", "shapoorji-demo")

    print(f"Space: {space_id}")
    print(f"Project: {project_name}")
    print()

    for name, template in EVALUATORS.items():
        print(f"Evaluator: summarization-{name}")
        print(f"Template:\n{template}\n")
        print("---")

    print(
        "\nUse `ax evaluator create` or the Arize UI to create these as online evaluators."
    )
    print("Each evaluator should be configured to trigger on summarization traces.")


if __name__ == "__main__":
    main()
