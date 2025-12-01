#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dotenv import load_dotenv

from ipynb.income_tax_graph import graph_builder, vector_store, retriever

load_dotenv()


# In[2]:


from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str # 사용자 질문
    answer: str # 세율
    tax_base_equation: str #과세표준 계산 수식
    tax_deduction: str#공제액
    market_ratio: str #공정시장가액비율
    tax_base: str #과세표준 계산

graph_builder = StateGraph(AgentState)


# In[3]:


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name = 'real_estate_tax',
    persist_directory='./real_estate_tax_collection',
)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})


# In[4]:


query = '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?'


# In[5]:


from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = hub.pull('rlm/rag-prompt')

llm = ChatOpenAI(model='gpt-4o')
small_llm = ChatOpenAI(model='gpt-4o-mini')


# In[6]:


tax_base_retrieval_chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
)

tax_base_equation_prompt = ChatPromptTemplate.from_messages([
    ('system', '용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 부연설명 없이 수식만 리턴해주세요'),
    ('human', '{tax_base_equation_information}')
])

tax_base_equation_chain = (
    {'tax_base_equation_information': RunnablePassthrough()}
    | tax_base_equation_prompt
    | llm
    | StrOutputParser()
)

tax_base_chain = {'tax_base_equation_information': tax_base_retrieval_chain} | tax_base_equation_chain

def get_tax_base_equation(state: AgentState):
    tax_base_equation_question = '주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요'
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)

    return {'tax_base_equation': tax_base_equation}


# In[7]:


#{'tax_base_equation': '과세표준 = (주택의 공시가격 합계 - 공제 금액) × 공정시장가액비율'}


# In[8]:


tax_deduction_chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

def get_tax_deduction(state: AgentState):
    tax_deduction_question = '주택에 대한 종합부동산세 계산시공제금액을 알려주세요'
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return {'tax_deduction': tax_deduction}


# In[9]:


# get_tax_deduction({})
#{'tax_deduction': '주택에 대한 종합부동산세의 공제 금액은 1세대 1주택자는 12억 원, 법인 또는 법인으로 보는 단체는 6억 원, 그리고 그 외의 경우는 9억 원입니다.'}


# In[10]:


from langchain_community.tools import TavilySearchResults
from datetime import date

tavily_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)

tax_market_ratio_prompt = ChatPromptTemplate.from_messages([
    ('system', f'아래 정보를 기반으로 공정시장 가액비율을 계산해주세요\n\nContext:\n{{context}}'),
    ('human', '{query}')
])

def get_market_ratio(state: AgentState):
    query = f'오늘 날짜:({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇 %인가요?'
    context = tavily_search_tool.invoke(query)
    tax_market_ratio_chain = (
        tax_market_ratio_prompt
        | llm
        |StrOutputParser()
    )
    tax_market_ratio = tax_market_ratio_chain.invoke({'context': context, 'query': query})
    return {'market_ratio': tax_market_ratio}


# In[11]:


# get_market_ratio({})
#{'market_ratio': '2025년의 주택 공시가격 공정시장가액비율은 60%입니다.'}


# In[12]:


from langchain_core.prompts import PromptTemplate

tax_base_calculation_prompt = ChatPromptTemplate.from_messages([
    ('system',"""
    주어진 내용을 기반으로과세표준을 계산해주세요

    과세표준 계산 공식: {tax_base_equation}
    공제금액: {tax_deduction}
    공정시장가액비율: {market_ratio}
    사용자 주택 공시가격 정보: {query}"""),
    ('human', '사용자 주택 공시가격 정보: {query}')
])

def calculate_tax_base(state: AgentState):
    tax_base_equation = state['tax_base_equation']
    tax_deduction = state['tax_deduction']
    market_ratio = state['market_ratio']
    query = state['query']
    tax_base_calculation_chain = (
        tax_base_calculation_prompt
        | llm
        | StrOutputParser()
    )
    tax_base = tax_base_calculation_chain.invoke({
        'tax_base_equation': tax_base_equation,
        'tax_deduction': tax_deduction,
        'market_ratio': market_ratio,
        'query': query
    })
    print(f"tax_base == {tax_base}")
    return {'tax_base': tax_base}


# In[13]:


# In[15]:


tax_rate_calculation_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의질문에 대한 종합부동산세를 계산해주세요

    종합부동산세 세율:{context}'''),
    ('human', '''과세표준 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요

    과세표준: {tax_base}
    주택 수: {query}
    ''')
])

def calculate_tax_rate(state: AgentState):
    query = state['query']
    tax_base = state['tax_base']
    context = retriever.invoke(query)
    tax_rate_chain = (
        tax_rate_calculation_prompt
        | llm
        | StrOutputParser()
    )
    tax_rate = tax_rate_chain.invoke({
        'context': context,
        'tax_base': tax_base,
        'query': query
    })
    print(f"tax_rate == {tax_rate}")
    return {'tax_rate': tax_rate}


# In[16]:
# In[2
# In[22]:


graph_builder.add_node('get_tax_base_equation', get_tax_base_equation)
graph_builder.add_node('get_tax_deduction', get_tax_deduction)
graph_builder.add_node('calculate_tax_rate', calculate_tax_rate)
graph_builder.add_node('get_market_ratio', get_market_ratio)
graph_builder.add_node('calculate_tax_base', calculate_tax_base)


# In[23]:


from langgraph.graph import START, END
graph_builder.add_edge(START, 'get_tax_base_equation')
graph_builder.add_edge(START, 'get_tax_deduction')
graph_builder.add_edge(START, 'get_market_ratio')
graph_builder.add_edge('get_tax_base_equation', 'calculate_tax_base')
graph_builder.add_edge('get_tax_deduction', 'calculate_tax_base')
graph_builder.add_edge('get_market_ratio', 'calculate_tax_base')
graph_builder.add_edge('calculate_tax_base', 'calculate_tax_rate')
graph_builder.add_edge('calculate_tax_rate', END)


# In[24]:


graph = graph_builder.compile()




