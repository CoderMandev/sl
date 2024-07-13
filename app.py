#IMPORTS
import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

LLAMA_API = "LL-isamupZquFN0kRXdn3w2Gh9Qdvy7ECoO66AX4JHD6Sm9NHrg8BbxWZFXLIzEKqHN"
LANGSMITH_API = "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"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API
os.environ["LANGCHAIN_PROJECT"] = "TASK_WORK"


#DEFINING THE NECESSARY DATA FOR THE MODEL
TASK_TYPES = """The following are the classifications of tasks:

GQ - This means general question, where the query is just a general question about task management
WQ - This means website question, where the query is about website features, or how to navigate the website
CTQ - This means create task query, where the query has instructed the model to create a query, and provided its details 
UTQ - This means update task query, where the query has instructed the model to update an already existing query with new details
DTQ - This means delete task query, where the query has instructed the model to delete an already existing query
MQ - This means mixed query, where the query has multiple of the same or different query types (GQ, WQ, CTQ, UTQ, DTQ)

"""
TASK_TEMPLATE = """A task has the following fields:
    Title - Summary of the task
    Description - Very brief description of the task
    Date - Date by which the task should be complete
    Time - Time by which the task should be complete
"""
iTASK_TYPES = """The following are the classifications of tasks:

GQ - This means general question, where the query is just a general question about task management
WQ - This means website question, where the query is about website features, or how to navigate the website
CTQ - This means create task query, where the query has instructed the model to create a query, and provided its details 
UTQ - This means update task query, where the query has instructed the model to update an already existing query with new details
DTQ - This means delete task query, where the query has instructed the model to delete an already existing query

"""


#MODEL 1 -- DETERMINE THE QUERY TYPE (STEP 1)
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
window_memory =ConversationBufferWindowMemory(k=10)

model = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    max_tokens=2048,
   
    
)

task_determination_template = PromptTemplate(
    template="""system
    You are a model designed to classify the type of tasks to run from and input query. \\n

    {TASK_TYPES} \\n\\n

    Given the user's query, determing the query type. \\n\\n

    Return a JSON object that contains two keys: 'question' and 'q_type'. \\n 
    
    'question' field contains the query corrected and simplified. \\n

    'q_type' field contains the query type determined. \\n

    Refer to the following examples\\n

    User: Hey assistant, schedule a meeting for me at 9:00 pm today \\n
    Result: 
    {{"question":"schedule meeting for 9:00pm today",
      "q_type":"CTQ"}} \\n

    User: I missed my meeting yesterday. Kindly delete that task for me\\n
    Result:
    {{"question":"delete missed meeting yesterday",
      "q_type":"DTQ"}} \\n

    User: How do I create a task?\\n
    Result: 
    {{"question":"Find out how to create task",
      "q_type":"WQ"}} \\n

    User: Is it a good idea to create a task for late night or early morning?\\n
    Result: 
    {{"question":"Better option: create task for late night or early morning",
      "q_type":"GQ"}} \\n
    
    User: Hi. Kindly schedule a meeting for me at 10:00 am tomorrow and move the meeting I have tonight from 8:00pm to 9:00pm\\n
    Result: 
    {{"question":"Schedule a meeting at 10:00 amd tomorrow, and modify meeting time for today from 8:00pm to 9:00 pm",
      "q_type":"MQ"}} \\n
    
    User: I want to watch a basketball game 8 hours from now. I want to have gonn shopping 2 hours before then \\n  
    Result:  
    {{'question': 'Schedule to watch basketball in 8 hours and schedule shopping 2 hours before the basketball game',
      'q_type': 'MQ'}} \\n \\n
    
    user
    Question: {query} \\n
    assistant""",
    input_variables=["query","TASK_TYPES"],
)

planning_route = task_determination_template | model | JsonOutputParser()

def determine_query_type(state):
    print('Determining the task type')
    result = planning_route.invoke(
        {"query": state["query"], 
         "TASK_TYPES": TASK_TYPES}
         )
    if result["q_type"] == "MQ":
        return {"ref_query":result["question"],"q_type":result["q_type"],"multiple":'True'}
    return {"ref_query":result["question"],"q_type":result["q_type"]}


#TASK CREATION TEMPLATE
task_detail_picker = PromptTemplate(
    template="""system
    You are a model designed to extract specific details from a task query\\n

    {TASK_TEMPLATE} \\n\\n

    Given the query, extract the information above. \\n\\n

    Return a JSON object that contains four keys as per the classification of details.\\n 
    
    If a detail is missing, instead write 'None'\\n
    
    If the query has implicitly define a date or time, use the following information to fill the fields.\\n
    Current time [DAY,DATE,TIME] {date}. The date is in year-month-day format, and time in 24h system. \\n

    If the query has not implicitly or explicitly defined a date or time, the default time for date is today, and time is 11:59pm. However, be smart. For example, you can't schedule a meeting for 10:00am today, if it is already 5pm; it can only be tomorrow 10:00am.\\n

    The title and description can only be missing when no description is in the query. Otherwise, always craft a suitable title and description\\n

    Refer to the following examples\\n

    
    User: Schedule meeting for 9:00pm today\\n
    
    Result: 
    {{"Title":"Meeting",
      "Description":"You have a meeting today",
      "Date":"2024-07-12",
      "Time":"9:00pm"}} \\n

    Notice below that two-weeks time is calculated from the current date:\\n
    Date: {{'date': ['Friday', '2022-11-13', '16:53:01']}}
    User: Son's basketball game in two weeks!\\n
    
    Result: 
    {{"Title":"Son's BasketBall Game",
      "Description":"Don't miss your son's basketball game!",
      "Date":"2022-11-25",
      "Time":"11:59pm"}} \\n

    Notice for this example, it is already 4 pm. Therefore, the meeting can only be tomorrow as default\\n
    Date: {{'date': ['Friday', '2023-06-1', '16:53:01']}}
    User: Baby shower 10 o'clock\\n
    
    Result: 
    {{"Title":"Baby shower",
      "Description":"Attend a baby shower",
      "Date":"2023-06-1",
      "Time":"10:00am"}} \\n
    
    user
    Question: {query} \\n
    assistant""",
    input_variables=["query","TASK_TEMPLATE","date"],
)

task_creating_route = task_detail_picker | model | JsonOutputParser()

def task_creator(state):
    if state['multiple'] == 'True':
        state['task_output'].pop(state["num_tasks"]-1)

    result = task_creating_route.invoke(
        {"query": state["ref_query"], 
         "TASK_TEMPLATE": TASK_TEMPLATE,
         "date":state["date"]}
         )
    return {"task_output":[{"CTQ":result},*[x for x in state["task_output"] if x is not None]]}
    



#TASK MODIFICATION
task_detail_highlighter = PromptTemplate(
    template="""system
    You are a model designed to extract specific details from a task query\\n

    {TASK_TEMPLATE} \\n\\n

    Given the query, extract the information above that is outlined in the query. \\n\\n

    Return a JSON object that contains an arbitrary number of keys, Depending on how many details you could extract.\\n 
    
    If a detail is missing, do not add it to the JSON object\\n
    
    If a title is to be changed, then it is likely that the description will change as well; and vice versa.\\n

    If the query has implicitly define a date or time, use the following information to fill the fields.\\n
    Current time [DAY,DATE,TIME] {date}. The date is in year-month-day format, and time in 24h system. \\n

    If the query has not implicitly or explicitly defined a date or time, do not add it to the JSON object \\n

    Refer to the following examples\\n
    Date: ['Friday', '2022-08-33', '11:03:04']
    User: Meeting for 9:00am postponed to 1:00pm\\n
    
    Result: 
    {{"Time":"1:00pm"}} \\n

    Date: ['Friday', '2024-01-3', '18:03:04']
    User: Change title from Date with Abigael to date with Abigail\\n
    
    Result: 
    {{"Title":"Date with Abigail,
      "Description":"You have a date with Abigail",}} \\n

    Date: ['Friday', '2024-02-19', '5:03:04']
    User: Change baby shower from today to tomorrow\\n
    
    Result: 
    {{"Date":"2024-02-20"}} \\n

    Notice below, that the current date was Tuesday. Next week Wednesday is 8 days later. This date is 2024-07-2:\\n
    Date: ['Tuesday', '2024-06-25', '10:03:04']
    User: Change the date with Abby from Wednesday to next week Wednesday.\\n
    
    Result: 
    {{"Date":"2024-07-2"}} \\n
    

    user
    Question: {query} \\n
    assistant""",
    input_variables=["query","TASK_TEMPLATE","date"],
)

task_updating_route = task_detail_highlighter | model | JsonOutputParser()

def task_updator(state):
    if state['multiple'] == 'True':
        state['task_output'].pop(state["num_tasks"]-1)
    print('Currently Updating the task')
    result = task_updating_route.invoke(
        {"query": state["ref_query"], 
         "TASK_TEMPLATE": TASK_TEMPLATE,
         "date":state["date"]}
         )
    return {"task_output":[{"UTQ":result},*[x for x in state["task_output"] if x is not None]]}



#GENERAL QUESTION
gq_answer_template = PromptTemplate(
    template="""system
    Give a general answer to a question.Be brief, so as to reduce expenses. Have a friendly tone\\n
    
    Here is the current date incase it is relevant: Current time [DAY,DATE,TIME] {date}. The date is in year-month-day format, and time in 24h system.
    user
    Question: {query} \\n
    assistant""",
    input_variables=["query", "date"],
)
gq_answering_route = gq_answer_template | model | StrOutputParser()

def general_question(state):
    if state['multiple'] == 'True':
        state['task_output'].pop(state["num_tasks"]-1)
    print('Currently answering a general question')
    result = gq_answering_route.invoke(
        {"query": state["ref_query"],
        "date": state["date"]}
         )

    return {"task_output":[{"GQ":result},*[x for x in state["task_output"] if x is not None]]}


#WEBSITE QUESTION
wq_answer_template = PromptTemplate(
    template="""system
    Give a general answer to a question.Be brief, so as to reduce expenses. Have a friendly tone\\n

    Here is the current date incase it is relevant: Current time [DAY,DATE,TIME] {date}. The date is in year-month-day format, and time in 24h system.
    
    user
    Question: {query} \\n
    assistant""",
    input_variables=["query","date"],
)
wq_answering_route = wq_answer_template | model | StrOutputParser()

def website_question(state):
    if state['multiple'] == 'True':
        state['task_output'].pop(state["num_tasks"]-1)
    print('Currently answering a website question')
    result = wq_answering_route.invoke(
        {"query": state["ref_query"],
         "date":state["date"]}
         )
    return {"task_output":[{"WQ":result},*[x for x in state["task_output"] if x is not None]]}

#TASK DELETION
def task_deletion(state):
    if state['multiple'] == 'True':
        state['task_output'].pop(state["num_tasks"]-1)
    print('Deleting a task!')
    result = "Cant do this yet"
    return {"task_output":[{"DTQ":result},*[x for x in state["task_output"] if x is not None]]}



#MULTIPLE QUERIES!
task_breakdowner = PromptTemplate(
    template="""system
    You are a model designed to identify the types of tasks in a given query\\n

    {iTASK_TYPES} \\n\\n

    Given the query, break it down to task and task type. \\n\\n

    Return a JSON object that contains an arbitrary number of keys, Depending on how many tasks you could extract.\\n 

    Each key should be a task type, and the value should be a list of tasks that belong to that task type\\n

    Refer to the following examples to learn how data should be structured. The values for keys should be made wrapped into a list\\n \\n

    User: Meeting for 9:00am postponed to 1:00pm. Baby shower to attend tomorrow\\n
    Result: 
    {{"UTQ":["Meeting for 9:00am postponed to 1:00pm"],
      "CTQ":["Baby shower to attend tomorrow"]}} \\n

    User: Change title from Date with Abigael to date with Abigail. Then, reschedule cooking classes from 9:00am tomorrow to 1:00pm tomorrow\\n
    Result: 
    {{"UTQ":["Change title from Date with Abigael to date with Abigail",
            "reschedule cooking classes from 9:00am tomorrow to 1:00pm tomorrow"]}} \\n

    User: Read Bible at 9:00am, Take shower at 10:00am, Leave for work at 11:00am. Postpone meeting time to 2:00pm\\n
    Result: 
    {{"CTQ":["Read Bible at 9:00am",
            "Take shower at 10:00am",
            "Leave for work at 11:00am."],
      "UTQ":["Postpone meeting time to 2:00pm"]}} \\n

    User: No longer have a date with Abby on Wednesday, 2:00pm. Schedule movies on that day at 3:00pm.\\n
    Result: 
    {{"DTQ":["No longer have a date with Abby on Wednesday"],
      "CTQ":["Schedule movies on Wednesday at 3:00pm."]}} \\n

    User: No longer have meeting today at 8:00pm. I have a class at 10:00pm.\\n
    Result: 
    {{"DTQ": ["No longer have meeting today at 8:00pm"],
      "CTQ": ["I have a class today at 10:00pm"]}} \\n \\n

    Do take note that the same date or can be mentioned for two different tasks.If and only if this is so, then, when giving a response, make sure each task's detail has the time and/or date \\n
    Refer to the following examples to know how the object should be structured. Look at the following example:\\n
    
    User: Today I have a LOT of work. I have to prepare food for the kids by 8am, and still be at work by 10 pm. Then, I should submit work documents by 4:00pm. Thank God the meeting at 3:00pm was postponed to tomorrow!
    Result:
    {{"CTQ":["prepare food for the kids by 8 am today",
            "be at work by 10 pm today",
            "submit the work documents by 4:00pm today"],
      "UTQ": ["the meeting at 3:00pm today was postponed to tomorrow"]}} \\n
    Notice that today has been repeated four times. \\n \\n

    user
    Question: {query} \\n
    assistant""",
    input_variables=["query","iTASK_TYPES"],
)

mixed_task_route = task_breakdowner | model | JsonOutputParser()

def mixed_task_handler(state):
    print('Currently breaking down a mixed task')
    result_imp = mixed_task_route.invoke(
        {"query": state["ref_query"], 
         "iTASK_TYPES": iTASK_TYPES}
    )
    result = handle_mq(result_imp)
    result.append({"DONE":"NECESSARY STOP SEQUENCE"})
    state.update({"task_output":result})
    return state


def handle_mq(mq_detail):
      print(mq_detail)
      return_list = []
      for key, val in enumerate(mq_detail):
            for item in mq_detail[val]:
                  return_list.append({val:item})
      return return_list

statue = {'UTQ': ['Joan rescheduled the meeting today from 8pm to 9pm'], 'DTQ': ['cancel my zoom call for 10 pm']}


def handle_mq(mq_detail):
      print(mq_detail)
      return_list = []
      for key, val in enumerate(mq_detail):
            for item in mq_detail[val]:
                  return_list.append({val:item})
      return return_list

handle_mq(statue)

mixed_task_handler({
        'query':"",
        'ref_query':"Joan rescheduled the meeting today from 8pm to 9pm. That girl is annoying. I will have to cancel my zoom call fro 10 pm",
        'q_type': "TBD",
        'task_output': [],
        'multiple': False,
        'num_tasks':0,
        'date':['Saturday', '2024-07-13', '12:51:31']
        })

#DETERMINE NEXT FUNCTION (NECESSARY FOR OPERATION)
def determine_next_function(state):
    cur_task = state['num_tasks']
    task_dict = state['task_output'][cur_task]
    new_q_type = list(task_dict.keys())[0]
    new_ref_query = task_dict[new_q_type]
    state.update({"ref_query":new_ref_query,"q_type":new_q_type})
    state["num_tasks"] += 1 
    return state
  
            

#GET DAY, DATE AND TIME
from datetime import datetime
def get_day_date_time(state):
    current_datetime = datetime.now()
    date = current_datetime.strftime("%Y-%m-%d")
    time = current_datetime.strftime("%H:%M:%S")
    day = current_datetime.strftime("%A")
    return {'date':[day,date,time]}
statuere = {}
get_day_date_time(statuere)    


#COMPILE YOUR ANSWER
def compile_answer(state):
    if state["multiple"] == 'True':
        state["q_type"] = "MQ"
        state['task_output'].pop()
    return state



#CHECK WHETHER QUERY IS AN MQ (NECESSARY FOR OPERATION)
def check_if_multiple(state):
    if state["multiple"] == 'True':
        return "multiple"
    else:
        return "single"


#CHECKING STATE (WHICH IS THE NEXT FUNCTION TO CALL) 
def check_state(state):
    if state["multiple"] == 'True':
        if state["num_tasks"] >= len(state["task_output"]) and len(state["task_output"]) > 0:
            return "DONE"
    return state["q_type"]



#NODES AND CONNECTION (!!!!)
#Create and configure the workflow
from typing_extensions import TypedDict

class WorkflowState(TypedDict):
    query: str
    ref_query: str
    q_type: str
    task_output: list
    multiple: str
    num_tasks: int
    date: list

    
from langgraph.graph import END, StateGraph
workflow = StateGraph(WorkflowState)

workflow.add_node("CTQ", task_creator)
workflow.add_node("UTQ", task_updator)
workflow.add_node("DTQ", task_deletion)
workflow.add_node("WQ", website_question)
workflow.add_node("GQ", general_question)
workflow.add_node("MQ", mixed_task_handler)
workflow.add_node("QRYTYP", determine_query_type)
workflow.add_node("GETTYM", get_day_date_time)
workflow.add_node("COMPANS", compile_answer)
workflow.add_node("DET_NXT_IN_MULT", determine_next_function)

workflow.set_entry_point("QRYTYP")

workflow.add_edge("QRYTYP","GETTYM")
workflow.add_conditional_edges(
    "GETTYM",
    check_state,
    {
        "CTQ": "CTQ",
        "UTQ": "UTQ",
        "DTQ": "DTQ",
        "WQ": "WQ",
        "GQ": "GQ",
        "MQ": "MQ",
    }
    )

workflow.add_edge("MQ","DET_NXT_IN_MULT")
nodelist = ["CTQ","UTQ","DTQ","WQ","GQ"]
def add_cedges(nodelist):
    for node in nodelist:
        workflow.add_conditional_edges(
            node,
            check_if_multiple,
            {
                "single":"COMPANS",
                "multiple":"DET_NXT_IN_MULT"
            }
            )
workflow.add_conditional_edges(
    "DET_NXT_IN_MULT",
    check_state,
    {
        "CTQ": "CTQ",
        "UTQ": "UTQ",
        "DTQ": "DTQ",
        "WQ": "WQ",
        "GQ": "GQ",
        "DONE":"COMPANS"
    }
    )


add_cedges(nodelist)


workflow.add_edge("COMPANS",END)




app = workflow.compile()



#TEST IT OUT NOW!
inputs = {
        'query':input('Question: '),
        'ref_query':"",
        'q_type': "TBD",
        'task_output': [],
        'multiple': False,
        'num_tasks':0,
        'date':[]
        }
result = app.invoke(inputs)
print(result)