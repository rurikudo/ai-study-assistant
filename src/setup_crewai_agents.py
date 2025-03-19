from crewai import Agent, Task, Crew, Process
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from crewai_tools import LlamaIndexTool
from llama_index.llms.openai import OpenAI
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

class TeachingAssistant:
    def __init__(self, data_dir: str, llm_model: str = "gpt-3.5-turbo"):
        self.llm_model = OpenAI(model=llm_model)
        self.root_data_dir = data_dir
        self.query_tool = self.prepare_query_tool()
        self.knowledge_expert = self.setup_knowledge_expert()
        self.advisor = self.setup_advisor()
        self.knowledge_expert_task = self.setup_knowledge_expert_task()
        self.advisor_task = self.setup_advisor_task()
        self.web_designer = self.setup_web_designer()
        self.web_designer_task = self.setup_web_designer_task()
        self.crew = self.ta_crew()

    def prepare_query_tool(self):
        documents_record = SimpleDirectoryReader(os.path.join(self.root_data_dir, "Record")).load_data(".pdf")
        documents_textbook = SimpleDirectoryReader(os.path.join(self.root_data_dir, "Textbook")).load_data(".pdf")
        documents = documents_record + documents_textbook
        Settings.chunk_size = 1024
        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        vector_index = VectorStoreIndex(nodes)
        vector_query_engine = vector_index.as_query_engine()
        return LlamaIndexTool.from_query_engine(
                        vector_query_engine,
                        name="Lecture Record Query Tool",
                        description="Use this tool to lookup information from the lecture recording and textbook",
                    )

    def setup_knowledge_expert(self) -> Agent:
        '''
        This agent uses the RagTool to answer questions.
        '''
        return Agent(
            role="Knowledge Expert",
            goal="Provide accurate and reliable information on a topic",
            backstory="An expert who knows a lot about a topic",
            verbose=True,
            tools=[self.query_tool],
        )
    
    def setup_knowledge_expert_task(self) -> Task:
        '''
        This task is for the knowledge expert agent.
        '''
        return Task(
            name="Knowledge Expert Task",
            description="Summarize information related to a question from a student: {question}",
            expected_output= f"A collection of bullet points detailing the major insights about a given question from a student.",
            agent=self.knowledge_expert,
            verbose=True,
        )

    def setup_advisor(self) -> Agent:
        '''
        This agent answers questions from students.
        '''
        return Agent(
            role="Advisor",
            goal="Answer a question from a student clearly and appropriately, with sufficient information",
            backstory="An experienced advisor working at a university",
            verbose=True,
        )

    def setup_advisor_task(self) -> Task:
        return Task(
            name="Advisor Task",
            description="Answer a question from a student clearly and appropriately, with sufficient information using the information from knowledge expert: {question}",
            expected_output= f"A clear main response to a given question from a student and a detailed list of bullet points explaining the reasons.",
            agent=self.advisor,
            context=[self.knowledge_expert_task],
            verbose=True,
        )
    
    def setup_web_designer(self) -> Agent:
        '''
        This agent converts the answer into web format.
        '''
        return Agent(
            role="Web Designer",
            goal="Convert the answer into a HTML format.",
            backstory="An experienced web designer who knows how to create web pages.",
            verbose=True,
        )

    def setup_web_designer_task(self) -> Task:
        return Task(
            name="Web Designer Task",
            description="Convert the answer into a HTML format",
            expected_output= f"A html content with the answers to the question from a student. The content can be placed within the <div> tag. The content must be simple and easy to read.",
            agent=self.web_designer,
            context=[self.advisor_task],
            verbose=True,
        )
    
    def ta_crew(self) -> Crew:
        return Crew(
            name="Study Assistant: Answering Questions",
            description="A crew that answers questions from students",
            agents=[self.knowledge_expert, self.advisor, self.web_designer],
            tasks=[self.knowledge_expert_task, self.advisor_task, self.web_designer_task],
            verbose=True,
            process=Process.sequential,
            # process=Process.hierarchical,
            # manager_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"), 
            )


class QuestionSetter:
    def __init__(self, question_example_file, main_material_file):
        self.question_source = PDFKnowledgeSource(file_paths=[question_example_file])
        self.main_material_source = PDFKnowledgeSource(file_paths=[main_material_file])
        self.topic_selector = self.setup_topic_selector()
        self.question_generator = self.setup_question_generator()
        self.topic_selector_task = self.setup_topic_selector_task()
        self.question_generator_task = self.setup_question_generator_task()
        self.web_designer = self.setup_web_designer()
        self.web_designer_task = self.setup_web_designer_task()
        self.crew = self.qs_crew()

    def setup_question_generator(self) -> Agent:
        '''
        This agent generates questions for students to help them learn.
        It uses the RagTool to generate questions.
        '''
        return Agent(
            role="Question_Generator",
            goal="Generate questions for students to help them learn. Referring to past questions to generate new questions.",
            backstory="An experienced question generator who knows how to create questions",
            knowledge_sources=[self.question_source],
            verbose=True,
        )

    def setup_question_generator_task(self) -> Task:
        return Task(
            name="Question Generator Task",
            description="Generate questions for students to help them prepare for exams using the information from the topic selector and past questions.",
            expected_output= f"A list of 5 questions and the answers for students to help them learn.",
            agent=self.question_generator,
            context=[self.topic_selector_task],
            verbose=True,
        )
    
    def setup_topic_selector(self) -> Agent:
        '''
        This agent select 5 topics randomly.
        '''
        return Agent(
            role="Topic_Selector",
            goal="Select five topics randomly to generate practice questions for students",
            backstory="An experienced teacher who is very good at selecting good topics from main material for exam preparation",
            knowledge_sources=[self.main_material_source],
            verbose=True,
        )

    def setup_topic_selector_task(self) -> Task:
        return Task(
            name="Topic Selector Task",
            description="Select five topics randomly to generate practice questions for exam preparation.",
            expected_output= f"A list of five topics and the important points of those topics.",
            agent=self.topic_selector,
            verbose=True,
        )
    
    def setup_web_designer(self) -> Agent:
        '''
        This agent converts the questions into web format.
        '''
        return Agent(
            role="Web Designer",
            goal="Convert the questions and answers into a HTML format.",
            backstory="An experienced web designer who knows how to create web pages.",
            verbose=True,
        )

    def setup_web_designer_task(self) -> Task:
        return Task(
            name="Web Designer Task",
            description="Convert the questions into a HTML format",
            expected_output= f"A html content with the questions and answers for students to learn. The content can be placed within the <div> tag. The content must include only five questions and their answers.",
            agent=self.web_designer,
            context=[self.question_generator_task],
            verbose=True,
        )
    
    def qs_crew(self) -> Crew:
        return Crew(
            name="Question Setter",
            agents=[self.topic_selector, self.question_generator, self.web_designer],
            tasks=[self.topic_selector_task, self.question_generator_task, self.web_designer_task],
            verbose=True,
            knowledge_sources=[self.question_source, self.main_material_source],
            process=Process.sequential,
            # process=Process.hierarchical,
            # manager_llm=ChatOpenAI(temperature=0, model="gpt-4"), 
            # manager_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        )
