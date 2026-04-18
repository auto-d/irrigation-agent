"""
Scratch-built agent framework to demonstrate the underlying abstractions

NOTE: Backend, Tool, Agent classes refactored from auto-d/voice-agent/agent.py
"""

from openai import OpenAI 
from copy import deepcopy
import json 

class Backend(): 
    """
    Class to abstract a backend, allows us to use one shape and rotate backend LLMs
    (to a degree) without changing agent logic. This can become an abstract base as 
    needed. 
    """

    def __init__(self, api_key, model="gpt-5.4", reasoning={"effort": "low"}): 
        """
        Create a backend model abstraction that simplifies text completions
        """
        self.api_key = api_key 
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.reasoning = reasoning

    def clone(self):
        """
        Create a deep copy of this instance 
        """
        new = Backend(self.api_key, self.model, self.reasoning)
        return new 

    def send(self, messages, instructions=None, tools=None): 
        """
        Send provided messages to the backend 
        """
        response = self.client.responses.create(
            model=self.model,
            #instructions=None, 
            input=messages, 
            reasoning=self.reasoning,
            tools=tools
        )

        return response
    
    def unpack_response(self, response, include_reasoning=False): 
        """
        Unpack and validate response, return plain language and any tool calls as arrays.
        """

        text = [] 
        tool_calls = []
        for r in response.output:
            match r.type: 
                case "reasoning":
                    if include_reasoning: 
                        text.append(r.summary)
                case "message": 
                    for c in r.content: 
                        match c.type: 
                            case "output_text": 
                                text.append(c.text)
                            case "function_call":
                                tool_calls.append(c)       
                            case "tool_call":
                                raise NotImplementedError("Need to implement support for newer function calling semantics!")
                            case _: 
                                raise ValueError(f"Unexpected message type received: {c.type}")
                case "function_call": 
                    tool_calls.append(r)
                case "tool_call":
                    raise NotImplementedError("Need to implement support for newer function calling semantics!")
                case _: 
                    raise ValueError(f"Unexpected response type received: {r.type}")
                
        return text, tool_calls 

class Memory(): 
    """
    Chat history with automatic summarization on demand
    """
    def __init__(self, backend=None): 
        self.backend = backend
        self.messages = []

    def clone(self):
        """
        Create a duplicate of this instance
        """
        new = Memory()
        new.backend = self.backend.clone() 
        new.messages = deepcopy(self.messages)
        return new 
    
    def append_developer(self, messages=[]): 
        """
        Add some new developer/system messages... 
        """        
        new = [ {"role": "developer", "content": msg} for msg in messages ]
        self.messages.extend(new)

    def append_user(self, messages=[]): 
        """
        Add some new messages
        """        
        new = [ {"role": "user", "content": msg} for msg in messages ]
        self.messages.extend(new)

    def append_assistant(self, messages=[]): 
        """
        Add some new messages
        """        
        new = [ {"role": "assistant", "content": msg} for msg in messages ]
        self.messages.extend(new)

    def append_response_output(self, responses=[]):
        """
        Append raw responses output... this is a bit confusing, the old completions 
        API had a format we were using prior, but the new Responses API seems to just 
        want these items reflected back when we're managing the message history.

        NOTE: Reflection syntax with help from gpt-5.4
        """
        for response in responses:
            if isinstance(response, dict):
                payload = deepcopy(response)
            elif hasattr(response, "model_dump"):
                payload = response.model_dump()
            else:
                payload = deepcopy(response.__dict__)

            # TODO: this is a workaround, we are including status for our telemetry but 
            # can't emit this to openAI... needs to be refactored. 
            if isinstance(payload, dict) and "status" in payload:
                del payload["status"]

            self.messages.append(payload)

    def append_tool_results(self, results=[]): 
        """
        Add tool call results in Responses API `function_call_output` form.
        """
        new = [
            {
                "type": "function_call_output",
                "call_id": result["call_id"],
                "output": json.dumps(result["result"]),
            }
            for result in results
        ]
        self.messages.extend(new)

    def summarize(self): 
        """
        """
        if self.backend is None: 
            raise ValueError("Summarization requested but no backend configured!")
        
        #TODO summarize! clients will want this e.g. for capturing history without clogging future contexts
        raise NotImplementedError()
    
class Tool(): 
    """
    Abstract the notion of an LLM tool and provide some minimal validation 
    to catch dumb mistakes. 
    """
    MINIMAL = {
        "type": "function",
        "name": None,
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }

    def __init__(self): 
        """
        Initialize with a full openAI tool schema. Derived classes must set the `schema`
        property on their instance before initialization of the parent (this class)
        """
        #TODO: this implicit relationship of the `schema` property is a little dicey, refactor
        Tool.validate(Tool.MINIMAL, self.schema)

    @classmethod 
    def validate(cls, src, target): 
        """
        Confirm the second object is a subset of the first 
        """
        if isinstance(src, dict):
            for k,v in src.items(): 
                if k not in target.keys(): 
                    raise ValueError(f"Provided tool schema is missing key '{k}'!")
                cls.validate(v, target[k])

    def validate_instance(self, instance): 
        """
        Validate a schema instance against the tool schema and return an object 
        representing the parsed call arguments

        NOTE: the OpenAI responses module returns tool calls as 
        ResponseFunctionToolCall objects. The properties of this object are 
        typically strings, not the objects we need to do real work. Hence the deserialization 
        going on in this method. Here's an example instance: 
        {
            arguments: '{"repository_url": "https://github.com/auto-d/water-quality","tasks": ["pull_repository","summarize_changes"]"file_edits": [],"commit_message": "","ci_job_names": []}', 
            call_id='call_8Q2gowqYYTedsf1gR4pB4I9F', 
            name='devops_tool', 
            type='function_call', 
            id='fc_043f0121b4c343140069d01ae10b84819188525ee393f37e3a', 
            namespace=None, 
            status='completed'
        }
        """        
        
        args = json.loads(instance.arguments) 
        print(f"Found {instance.type} {instance.name} with id {instance.id} (call ID: {instance.call_id})...")
        print("Arguments", args)

        #TODO: validate the object against the schema

        return args

    def run(self, instance, callback=None):
        """
        Invoke this tool given the parameters in instance, which must adhere
        to the tool schema. Derived classes should implement this to execute actual work. 

        Optionally provide a callback to handle the implementation yourself.

        Return an object here to simplify parsing by LLMs and derived classes. This serves
        also to ensure we have a rich output to mine in our telemetry which will be central 
        to evaluation. Schema: 
        { 
            "status": "ok" or similar string detailing result code 
            "call_id": <tool-call id
            "result" : { <tool-specific object> }
        }

        """
        raise NotImplementedError("run() method is an abstract method!")

class LlmAgent(): 
    """
    Abstract the generic notion of an LLM agent.
    """

    def __init__(self, backend, cwd='.', persist=False, dev_msgs=[]): 
        """
        Create a new agent instance 
        """
        #TODO: decide whether these should be instantiated internally, acepting 
        # as a param risks us sharing an instance which could result in some 
        # unnecessary sharing of I/O etc. 
        self.backend = backend 
        self.cwd = cwd
        self.persist = False
        self.memory = Memory(backend=backend)
        self.memory.append_developer(dev_msgs)

    def save(self, path): 
        """
        Persist agent configuration and state to disk 
        """
        pass 

    def load(self, path): 
        """
        Load an agent config off disk 
        """
        pass 

    def chat(self, messages, tools=[], trace=False):
        """
        Send a message to the agent with the provided user messages. Passed messages should 
        be text only and will be formatted internally before delivery to the backend. 

        Returns the response text from the backing model. 
        """

        self.memory.append_user(messages)

        response = None
        text = None 

        tool_schemas = [ x.schema for x in tools ]
        tool_map = { x.schema['name']: x for x in tools}

        # Pass the conversation to the backing LLM, iterating over and executing 
        # any returned tool calls until only a chat response come back (which will
        # perhaps be the dominant case). 
        while True: 
            if trace: 
                print("===============\n")
                print("Sending messages to backend...")
                print("Message history:\n",self.memory.messages)
                print("Tool descriptions:\n", tool_schemas)
                print("---------------\n")

            response = self.backend.send(self.memory.messages, tools=tool_schemas)
            text, tool_calls = self.backend.unpack_response(response)
            self.memory.append_response_output(response.output)
            
            if len(tool_calls) == 0: 
                break 

            for call in tool_calls: 
                tool = tool_map[call.name]
                result = tool.run(call)
                self.memory.append_tool_results([result])

        return text

    def run(self): 
        """
        Run this agent -- all agents currently run on the invoking thread's context, which works well for the 
        API-server as a proxy, but will degrade as the system grows. Eventually these will be durable and 
        freestanding but not today. :D
        """    
        raise NotImplementedError("Agent loop is currently external to this instance!")

class NeuralPlanner(LlmAgent): 
    """
    Execute planning duties leveraging an LLM backend

    TODO: adapt to the neural planning task
    """
    
    def __init__(self, backend, openai_api_key): 
        """
        Create an instance of our agent
        """
        super().__init__(backend=backend)

        self.cheap_backend = Backend(api_key=openai_api_key, model="gpt-5.4-mini", reasoning={ "effort": "low"} )
        
        # TODO: encode our actions as Tools and pass to the backend
        # self.fetch_tool = SummarizePageTool(backend=self.cheap_backend)

    def run(self, callback=None):
        """
        Run the agent 
        """
        
        prompt = f"Foo"
       
        # TODO: integrate planning! 

        #available_tools = [self.fetch_tool]
        #text = self.chat([prompt], tools=available_tools)
        #result = "\n".join(text).strip()

