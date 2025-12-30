from pydantic import BaseModel
from oai_utils.agent import AgentWrapper, AgentsSDKModel
from adapter.models.problems import QAProblem


class QAEvalResult(BaseModel):
    reason: str
    is_correct: bool


async def evaluate_qa(
    problem: QAProblem, answer: str, model: AgentsSDKModel = "gpt-5-mini"
) -> QAEvalResult:
    agent = AgentWrapper[QAEvalResult].create(
        name="qa_evaluator",
        instructions="""\
You are a Senior Technical Auditor and Software Engineer specialized in library-level system design and developer operations. 
Your task is to perform a rigorous comparative analysis between a "Ground Truth Answer" (GT) and a "Candidate's Answer" based on a provided "Question."

### Evaluation Criteria:
1. **Identity & Context Accuracy**: Does the Candidate's Answer identify the specific, correct targets, classes, or configuration flags used by the library?
2. **Logic & Mechanics**: Does the Candidate correctly explain 'how' the system works (e.g., variable precedence, build tools like maturin, or internal method composition)?
3. **Gap Analysis**: What specific technical nuances or "secret sauce" found in the GT are missing from the Candidate's Answer?
4. **Actionability**: Are the commands/codes provided by the Candidate literally correct, or are they generic "best guesses"?

### OUTPUT FORMAT:
You MUST respond with a valid JSON object that conforms to the following Pydantic-ready structure:
{
    "reason": "Explanation of why the answer is or is not correct."
    "is_correct": "boolean. True if the answer is semantically same with the ground truth, False otherwise.",
}""",
        model=model,
        output_type=QAEvalResult,
    )

    input_text = f"""\
Question: {problem.question}
Ground Truth Answer: {problem.answer}
Candidate's Answer: {answer}
"""
    ret = await agent.run(input=input_text)
    return ret.final_output()
