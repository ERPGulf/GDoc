# Create a prompt template to generate a question a end-user could have about a given context
initial_question_prompt_template = PromptTemplate(
    input_variables=["context"],
    template="""
    <Instructions>
    Here is some context:
    <context>
    {context}
    </context>

    Your task is to generate 1 question that can be answered using the provided context, following these rules:

    <rules>
    1. The question should make sense to humans even when read without the given context.
    2. The question should be fully answered from the given context.
    3. The question should be framed from a part of context that contains important information. It can also be from tables, code, etc.
    4. The answer to the question should not contain any links.
    5. The question should be of moderate difficulty.
    6. The question must be reasonable and must be understood and responded by humans.
    7. Do not use phrases like 'provided context', etc. in the question.
    8. Avoid framing questions using the word "and" that can be decomposed into more than one question.
    9. The question should not contain more than 10 words, make use of abbreviations wherever possible.
    </rules>

    To generate the question, first identify the most important or relevant part of the context. Then frame a question around that part that satisfies all the rules above.

    Output only the generated question with a "?" at the end, no other text or characters.
    </Instructions>
    
    """)
# Create a prompt template that takes into consideration the the question and generates an answer
answer_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    <Instructions>
    <Task>
    <role>You are an experienced QA Engineer for building large language model applications.</role>
    <task>It is your task to generate an answer to the following question <question>{question}</question> only based on the <context>{context}</context></task>
    The output should be only the answer generated from the context.

    <rules>
    1. Only use the given context as a source for generating the answer.
    2. Be as precise as possible with answering the question.
    3. Be concise in answering the question and only answer the question at hand rather than adding extra information.
    </rules>

    Only output the generated answer as a sentence. No extra characters.
    </Task>
    </Instructions>
    
    Assistant:""")
# To check whether an answer was correctly formulated by the large language model you get the relevant text passages from the documents used for answering the questions.
source_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Human:
    <Instructions>
    Here is the context:
    <context>
    {context}
    </context>

    Your task is to extract the relevant sentences from the given context that can potentially help answer the following question. You are not allowed to make any changes to the sentences from the context.

    <question>
    {question}
    </question>

    Output only the relevant sentences you found, one sentence per line, without any extra characters or explanations.
    </Instructions>
    Assistant:""")
# To generate a more versatile testing dataset you alternate the questions to see how your RAG systems performs against differently formulated of questions
question_compress_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    <Instructions>
    <role>You are an experienced linguistics expert for building testsets for large language model applications.</role>

    <task>It is your task to rewrite the following question in a more indirect and compressed form, following these rules:

    <rules>
    1. Make the question more indirect
    2. Make the question shorter
    3. Use abbreviations if possible
    </rules>

    <question>
    {question}
    </question>

    Your output should only be the rewritten question with a question mark "?" at the end. Do not provide any other explanation or text.
    </task>
    </Instructions>
    
    """)

# def initialize_claude():
#     return
# def 