#!/usr/bin/env python3
"""Milestone 5: Gradio interface for the Rutgers off-campus housing RAG system.

Run:
    python app.py
Then open http://localhost:7860
"""

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from generate import ask  # noqa: E402


def handle_query(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""
    result = ask(question.strip())
    sources_text = "\n".join(f"• {s}" for s in result["sources"])
    distances = ", ".join(str(c["distance"]) for c in result["chunks"])
    sources_display = f"{sources_text}\n\n[retrieval distances: {distances}]"
    return result["answer"], sources_display


EXAMPLE_QUESTIONS = [
    "What areas near Rutgers do students say are cheaper for off-campus housing?",
    "What platforms or apps do students recommend for finding housing near Rutgers?",
    "What safety advice do students give for choosing off-campus housing?",
    "How expensive is it to live off campus around Rutgers New Brunswick?",
    "What apartment complexes do grad students recommend near Rutgers NB?",
]

with gr.Blocks(title="Rutgers Off-Campus Housing Guide") as demo:
    gr.Markdown(
        "## Rutgers Off-Campus Housing — Unofficial Student Guide\n"
        "Ask anything about off-campus housing near Rutgers New Brunswick. "
        "Answers are grounded in student Reddit threads, official Rutgers resources, "
        "and rental listings — not general LLM knowledge."
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What neighborhoods near Rutgers are safest for grad students?",
                lines=2,
            )
        with gr.Column(scale=1):
            ask_btn = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(label="Answer", lines=8, interactive=False)
    sources_box = gr.Textbox(label="Retrieved from", lines=5, interactive=False)

    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=question_box,
        label="Example questions",
    )

    ask_btn.click(handle_query, inputs=question_box, outputs=[answer_box, sources_box])
    question_box.submit(handle_query, inputs=question_box, outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()
