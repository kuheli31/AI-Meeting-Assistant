# Actionable items, decisions, questions, and other items
# that require follow-up or further investigation.

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return GoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.3,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )


def build_chain(system_prompt: str):
    llm = get_llm()

    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}")
        ])
        | llm
        | StrOutputParser()
    )


def extract_actionable_items(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting analyst responsible for extracting
clear and genuinely actionable tasks from a meeting transcript.

Your job is to identify ONLY actions that were actually:
- Assigned
- Agreed upon
- Requested
- Committed to
- Clearly identified as requiring follow-up

Do NOT turn general discussion, suggestions, opinions, ideas, or
hypothetical possibilities into action items.

For every action item, extract:

- Task: A clear and specific description of what needs to be done.
- Owner: The person, group, or team responsible for completing it.
- Deadline: The exact date, time, or timeframe if explicitly mentioned.
  If no deadline is mentioned, write "Not specified".

Important rules:

1. Do not invent information.
2. Never guess an owner or deadline.
3. If the owner is not explicitly identifiable, write "Not specified".
4. If the deadline is not explicitly mentioned, write "Not specified".
5. Preserve important names, dates, numbers, deliverables,
   requirements, and constraints.
6. If multiple people or teams are explicitly responsible, include them.
7. If the same task appears multiple times, combine it into one item.
8. Do not include completed tasks unless the meeting explicitly
   assigns additional follow-up work.
9. Do not include statements such as "we should consider...",
   "maybe we can...", or "it would be good to..." unless the
   transcript clearly indicates that the task was accepted or assigned.
10. Focus on concrete work that someone needs to perform after or
    because of the meeting.
11. Keep each task concise but specific enough that someone could
    understand what needs to be done without reading the entire transcript.

Use exactly this format:

1. Task: ...
   Owner: ...
   Deadline: ...

2. Task: ...
   Owner: ...
   Deadline: ...

If there are no genuine action items, return exactly:

No action items found.

Return ONLY the numbered action-item list.
Do not add an introduction, explanation, conclusion, or commentary.
"""
    )

    return chain.invoke({
        "text": transcript
    })


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting analyst responsible for identifying
confirmed decisions from a meeting transcript.

Extract ONLY decisions that were actually:
- Made
- Agreed upon
- Approved
- Finalized
- Explicitly accepted

A decision must be supported by the transcript.

For every decision, clearly state:
- What was decided.
- The relevant context, only when necessary to understand the decision.

Important rules:

1. Do not confuse discussion with a decision.
2. Do not treat suggestions, proposals, opinions, possibilities,
   or questions as confirmed decisions.
3. Do not infer a decision that was not supported by the transcript.
4. If several alternatives were discussed, identify the selected
   option only if the transcript clearly indicates that it was chosen.
5. If something is still under discussion, pending approval, or
   conditional, do NOT present it as a final decision.
6. Preserve important names, dates, numbers, deadlines, selected
   options, requirements, and commitments.
7. Remove duplicate decisions even if the same decision is discussed
   multiple times.
8. Keep each decision concise and easy to understand.
9. Do not add information that is not present in the transcript.
10. Distinguish clearly between "what was discussed" and "what was decided".

Use exactly this format:

1. Decision: ...
   Context: ...

2. Decision: ...
   Context: ...

Only include the Context line when it provides useful information.

If there are no confirmed decisions, return exactly:

No key decisions found.

Return ONLY the numbered decision list.
Do not add an introduction, explanation, conclusion, or commentary.
"""
    )

    return chain.invoke({
        "text": transcript
    })


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting analyst responsible for identifying
unresolved questions, pending issues, and topics that require
follow-up or further investigation.

Extract ONLY issues that genuinely remain unresolved at the end
of the meeting.

Include:

- Questions that were explicitly asked but not answered.
- Issues requiring further investigation.
- Information that someone needs to confirm or obtain.
- Pending approvals or clarifications.
- Decisions that could not be finalized because additional
  information or discussion is required.
- Follow-up topics that clearly need to be revisited.
- Problems that remain unresolved and require additional work.

Important rules:

1. Do not include questions that were already answered.
2. Do not include rhetorical questions unless they require an
   actual response or follow-up.
3. Do not invent questions or unresolved issues.
4. Do not turn normal discussion into an open question.
5. Clearly distinguish between a confirmed decision and something
   that is still pending.
6. Preserve important names, dates, numbers, requirements,
   and relevant context.
7. If a responsible person or team is explicitly mentioned,
   include them.
8. If no owner is mentioned, do not guess one.
9. If the same unresolved issue appears multiple times, combine it.
10. Keep each item concise and easy to understand.
11. Focus on what still needs an answer, clarification,
    investigation, approval, or follow-up after the meeting.

Use exactly this format:

1. Question / Follow-up: ...
   Owner: ...
   Required follow-up: ...

2. Question / Follow-up: ...
   Owner: ...
   Required follow-up: ...

Only include Owner when a responsible person or team is explicitly
identified. Otherwise write:

Owner: Not specified

If there is no clearly unresolved question, pending issue,
or follow-up item, return exactly:

No open questions found.

Return ONLY the numbered list.
Do not add an introduction, explanation, conclusion, or commentary.
"""
    )

    return chain.invoke({
        "text": transcript
    })