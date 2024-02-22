import os
from textwrap import dedent
from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from dotenv import load_dotenv
import gradio as gr

# Initialize the search tools without needing API keys for demonstration
duckduckgo_search = DuckDuckGoSearchRun()
pubmed_search = PubmedQueryRun()
semanticscholar_search = SemanticScholarQueryRun()

class MedicalResearchProposalAgents:
    def __init__(self, api_key):
        self.api_key = api_key

    def intro_agent(self):
        return Agent(
            role='Strategic Introduction Architect',
            goal='To craft an engaging and informative introduction for the proposal.',
            tools=[duckduckgo_search, pubmed_search, semanticscholar_search],
            backstory='Expert in distilling complex medical issues into compelling narratives.',
            verbose=True,
            llm=ChatOpenAI(name="gpt-4", model="gpt-4-turbo-preview", temperature=0, max_tokens=2000, api_key=self.api_key)
        )

    def review_agent(self):
        return Agent(
            role='Expert Literature Analyst',
            goal='To perform a thorough review of existing literature.',
            tools=[duckduckgo_search, pubmed_search, semanticscholar_search],
            backstory='Adept at uncovering trends and gaps in medical research literature.',
            verbose=True,
            llm=ChatOpenAI(name="gpt-4", model="gpt-4-turbo-preview", temperature=0, max_tokens=2000, api_key=self.api_key)
        )

    def methodology_agent(self):
        return Agent(
            role='Research Methodology Design Expert',
            goal='To outline a clear and ethical methodology for the study.',
            tools=[duckduckgo_search, pubmed_search, semanticscholar_search],
            backstory='Combines expertise in research methods with a commitment to ethical standards.',
            verbose=True,
            llm=ChatOpenAI(name="gpt-4", model="gpt-4-turbo-preview", temperature=0, max_tokens=2000, api_key=self.api_key)
        )

    def statistics_agent(self):
        return Agent(
            role='Statistical Analysis Strategist',
            goal='To develop a comprehensive statistical analysis plan.',
            tools=[duckduckgo_search, pubmed_search, semanticscholar_search],
            backstory='Skilled in applying statistical methods to analyze complex datasets.',
            verbose=True,
            llm=ChatOpenAI(name="gpt-4", model="gpt-4-turbo-preview", temperature=0, max_tokens=2000, api_key=self.api_key)
        )

def generate_medical_research_proposal(openai_api_key, research_title):
    if not openai_api_key or not research_title:
        return "Please ensure both the OpenAI API Key and the Research Title are provided."

    agents_class = MedicalResearchProposalAgents(api_key=openai_api_key)

    intro_agent = agents_class.intro_agent()
    review_agent = agents_class.review_agent()
    methodology_agent = agents_class.methodology_agent()
    statistics_agent = agents_class.statistics_agent()

    introduction_task = Task(description=f"Write the INTRODUCTION section for a medical research proposal titled '{research_title}'. Write in 500 words. Write a list of the references that you used.", agent=intro_agent)
    literature_review_task = Task(description=f"Conduct a thorough literature search and literature review on the research title '{research_title}' and then write the LITERATURE REVIEW section for the medical research proposal. Write in 1000 words. Write a list of references that you used.", agent=review_agent)
    methodology_task = Task(description=f"Write the METHODOLOGY section for a medical research proposal titled '{research_title}' using similar previous studies as references. Write in 1000 words. Write a list of the references that you used.", agent=methodology_agent)
    statistics_task = Task(description=f"Write the STATISTICAL ANALYSIS section for a medical research proposal titled '{research_title}'. Write in 1000 words. Write a list of the reference tat you used.", agent=statistics_agent)

    writer_crew = Crew(
        agents=[intro_agent, review_agent, methodology_agent, statistics_agent],
        tasks=[introduction_task, literature_review_task, methodology_task, statistics_task],
        verbose=True,
        process=Process.sequential,
    )

    writer_crew.kickoff()

    final_proposal = f"""
    Introduction Section: 
    {introduction_task.output}
    Literature Review Section:
    {literature_review_task.output}
    Methodology Section:
    {methodology_task.output}
    Statistical Analysis Section:
    {statistics_task.output}
    """

    return final_proposal.strip()

iface = gr.Interface(
    fn=generate_medical_research_proposal,
    inputs=[
        gr.Textbox(label="OpenAI API Key", placeholder="Enter your OpenAI API Key here", type="password"),
        gr.Textbox(lines=2, label="Research Title", placeholder="Enter Research Title Here")
    ],
    outputs="text",
    title="Generate Medical Research Proposal",
    description="Generate a comprehensive proposal for your medical research. Ensure to provide a valid OpenAI API Key."
)

iface.launch()
