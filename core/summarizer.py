from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    return GoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.3,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=12000,
        chunk_overlap=3000
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert meeting analyst.

Analyze the meeting transcript carefully and create a clear, accurate
summary of the provided section.

Your summary must:
- Preserve the main topics, discussions, decisions, and important facts.
- Identify important conclusions or outcomes.
- Capture important problems, requirements, concerns, or questions.
- Preserve names, dates, numbers, deadlines, and responsibilities when present.
- Do not invent information or assume facts that are not stated.
- Remove repetition, filler words, greetings, and irrelevant conversation.
- Use simple, professional language that anyone can understand.
- Keep the summary concise but do not omit important information.
- Treat the text as part of a larger meeting, so preserve context that may
  be useful when the sections are combined later.

Return only the summary.
"""
        ),
        ("human", "{text}")
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    combined_summary = "\n\n".join(chunk_summaries)

    combine_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert meeting summarizer.

Create a single clear and easy-to-understand summary from the provided
meeting-section summaries.

Your goal is to make the entire meeting understandable even to someone
who did not attend it.

Follow these rules:
- Start with the overall purpose and main topic of the meeting.
- Organize the information logically rather than simply repeating the
  section summaries.
- Explain the most important discussions and conclusions.
- Clearly preserve decisions that were actually made.
- Preserve important action items, owners, deadlines, requirements,
  problems, and next steps when they are present.
- Remove duplicate information and conversational filler.
- Do not invent decisions, responsibilities, deadlines, or facts.
- If something is uncertain or was only discussed as a possibility,
  do not present it as a confirmed decision.
- Use simple, natural, professional English.
- Prefer concise paragraphs and bullet points where they improve clarity.
- Make the result useful as a quick reference after the meeting.

Return only the final meeting summary.
"""
        ),
        ("human", "{text}")
    ])

    combine_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x["text"]})
        | combine_prompt
        | llm
        | StrOutputParser()
    )

    return combine_chain.invoke({
        "text": combined_summary
    })


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You generate professional titles for meetings.

Read the meeting transcript and identify its main purpose and subject.

Create ONE short, specific, professional title.

Rules:
- Maximum 8 words.
- Clearly reflect the main subject of the meeting.
- Prefer meaningful nouns and keywords from the transcript.
- Avoid generic titles such as "Meeting Summary", "Team Meeting",
  "Discussion", or "Meeting Notes".
- Do not invent information.
- Do not use quotation marks.
- Return only the title and nothing else.
"""
        ),
        ("human", "{text}")
    ])

    title_chain = (
        title_prompt
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke({
        "text": transcript[:2000]
    })

