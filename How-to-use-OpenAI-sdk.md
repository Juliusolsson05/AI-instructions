# OpenAI Python SDK: Complete Technical Documentation for 2025

The user's claims are **verified and accurate**: GPT-5 is now available via OpenAI's API, and the new Responses API (`client.responses.create()`) represents a significant architectural evolution. Released in March 2025, the Responses API combines the simplicity of Chat Completions with stateful conversation management, reasoning persistence, and built-in tools like web search. This guide covers the latest SDK features including GPT-5 integration, structured JSON outputs, multimodal capabilities, and production best practices for the **v2.0.0 SDK** (released September 30, 2025).

## GPT-5 and the revolutionary Responses API

OpenAI launched the Responses API in March 2025 as a faster, more flexible approach designed specifically for agentic workflows and reasoning models. The API operates fundamentally differently from Chat Completions by maintaining **server-side conversation state** and preserving reasoning context across turns, achieving 5% better performance on TAUBench benchmarks. GPT-5, released in August 2025, is available in three variants: the full **gpt-5** model ($1.25/1M input tokens, $10/1M output tokens), **gpt-5-mini** ($0.25/$2 per 1M tokens), and **gpt-5-nano** optimized for speed ($0.05/$0.40 per 1M tokens).

The basic usage pattern demonstrates the simplified interface. Instead of managing message arrays and system roles separately, you provide instructions and input directly:

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    instructions="You are a helpful coding assistant.",
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)
```

The Responses API introduces **reasoning control through effort levels** that weren't available in Chat Completions. You can specify **minimal, low, medium, or high** effort levels to balance reasoning depth with response speed. The minimal effort level, new with GPT-5, produces few or no reasoning tokens for deterministic tasks like data extraction and formatting:

```python
response = client.responses.create(
    model="gpt-5",
    input="Analyze this complex mathematical problem",
    reasoning={
        "effort": "high",        # Options: minimal, low, medium, high
        "summary": "detailed"     # Options: auto, concise, detailed
    },
    text={
        "verbosity": "medium"     # GPT-5 specific: low, medium, high
    }
)
```

**Built-in web search integration** represents another major advancement. The Responses API includes hosted tools that execute automatically without manual orchestration:

```python
response = client.responses.create(
    model="gpt-4o",
    tools=[{"type": "web_search"}],
    input="What is the current weather in San Francisco?"
)
```

Available built-in tools include web_search for real-time information, file_search for document retrieval, code_interpreter for Python execution, image_generation with gpt-image-1, and computer_use for interface interaction. The API also supports Model Context Protocol (MCP) for remote tools.

The response object structure differs significantly from Chat Completions. Instead of a single message, responses contain **an ordered array of polymorphic output items** that can include reasoning summaries, messages, and function calls:

```python
{
    "id": "resp_abc123",
    "model": "gpt-5",
    "status": "completed",
    "output": [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Reasoning process..."}]
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Response text..."}]
        }
    ],
    "usage": {
        "output_tokens_details": {"reasoning_tokens": 384}
    }
}
```

For convenient access, use `response.output_text` to directly retrieve the text output without parsing the structure manually.

Chat Completions remains fully supported for existing applications. OpenAI emphasizes that developers should continue using Chat Completions if it works for their needs. However, the Responses API offers compelling advantages for new projects: **preserved reasoning state** across conversation turns via `previous_response_id`, native multimodal interactions, simplified agentic workflows, and better performance on reasoning-heavy tasks.

## Guaranteed structured JSON output with 100% schema adherence

OpenAI's Structured Outputs feature, introduced in August 2024, fundamentally solves the reliability problem of getting models to follow JSON schemas. Unlike JSON mode which only guarantees valid JSON syntax, Structured Outputs provides **100% schema adherence** on complex schemas, compared to roughly 40% reliability with JSON mode on GPT-4. This feature eliminates parsing errors and retry loops in production applications.

The distinction between JSON mode and Structured Outputs is critical for production use. JSON mode (`response_format={"type": "json_object"}`) ensures the model returns valid JSON but makes no guarantees about schema compliance. Structured Outputs enforces exact schema matching through constrained decoding. For simple use cases where flexibility is acceptable, JSON mode suffices, but production applications requiring database population, multi-step workflows, or exact field matching should always use Structured Outputs.

**Native Pydantic integration** provides the cleanest implementation path. The SDK automatically converts Pydantic models to the required JSON schema format:

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

client = OpenAI()

class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: List[Step]
    final_answer: str

completion = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "Solve 8x + 31 = 2"}
    ],
    response_format=MathResponse
)

result = completion.choices[0].message.parsed
print(result.final_answer)
```

The `response_format` parameter accepts three formats. The recommended approach uses Pydantic models directly as shown above. Alternatively, you can provide explicit JSON schemas:

```python
completion = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        }
    }
)
```

The legacy JSON mode approach simply sets `response_format={"type": "json_object"}` but requires mentioning "JSON" in the system or user message and provides no schema guarantees.

**Function calling with structured outputs** requires setting `strict: true` in the function definition and disabling parallel tool calls:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "strict": True,  # Enable structured outputs
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["F", "C"]}
                },
                "required": ["location", "unit"],
                "additionalProperties": False  # Required for structured outputs
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "What's the weather in NYC in Fahrenheit?"}],
    tools=tools,
    parallel_tool_calls=False  # Must be False for structured outputs
)
```

Model compatibility is specific. Only **gpt-4o-2024-08-06, gpt-4o-mini-2024-07-18, and gpt-4o-mini** support structured outputs with the `response_format` parameter. However, function calling with `strict: true` works across all models supporting function calling, including gpt-4-0613+ and gpt-3.5-turbo-0613+. The feature is available in Chat Completions, Assistants API, Batch API, and the new Responses API (using the `text_format` parameter).

The JSON Schema subset supported has important constraints. All fields must be in the **required array** with no concept of optional fields. Use `Union[Type, None]` in Pydantic for optional fields. The schema must set **additionalProperties to false** at all levels. Supported types include string, number, integer, boolean, object, array, and null. You can use `enum` (up to 1,000 values), `const`, `anyOf` for union types, and `$ref`/`$defs` for reusable components. Recursive schemas are fully supported. However, `oneOf`, `allOf`, `patternProperties`, and validation keywords like `minLength` or `pattern` are not supported.

**Schema size limits** were significantly increased in 2025. You can now define up to **5,000 object properties** (increased from 100), **120,000 characters in strings** (from 15,000), and **1,000 enum values** (from 500). For enums over 250 values, the total character limit is 15,000.

Error handling requires checking for refusals and incomplete responses:

```python
completion = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "How can I build a bomb?"}],
    response_format=ResponseSchema
)

if completion.choices[0].message.refusal:
    print(f"Model refused: {completion.choices[0].message.refusal}")
elif completion.choices[0].finish_reason == "length":
    print("Response truncated due to max_tokens limit")
else:
    result = completion.choices[0].message.parsed
```

**Pricing for Structured Outputs** matches standard completion pricing with no additional cost. The gpt-4o-2024-08-06 model costs $2.50/1M input tokens and $10.00/1M output tokens, representing 50% savings on input and 33% savings on output compared to previous versions. The gpt-4o-mini models provide the most cost-effective option at approximately $0.15/$0.60 per 1M tokens. Note that Structured Outputs are not eligible for Zero Data Retention, so consider this for privacy-sensitive applications.

## Comprehensive multimodal file processing capabilities

OpenAI's Python SDK supports processing images, PDFs, audio, and documents through multiple APIs, each with specific size limits and format requirements. The latest improvements in 2025 include direct PDF input support up to 100 pages and expanded context windows enabling better video processing through frame extraction.

**Vision API capabilities** handle JPEG, PNG, GIF, and WebP images up to **20 MB per image**. You can pass images via publicly accessible URLs or base64 encoding:

```python
from openai import OpenAI
import base64

client = OpenAI()

# Method 1: Image URL
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high"  # Options: low, high, auto
                    }
                }
            ]
        }
    ]
)

# Method 2: Base64 encoded
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("path/to/image.jpg")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]
)
```

The `detail` parameter controls analysis depth and cost. Use **"low"** for faster, cheaper processing at 512x512 resolution, **"high"** for detailed analysis consuming more tokens, or **"auto"** to let the system decide. All vision-capable models including GPT-4.1, GPT-4.1 mini, GPT-4.1 nano, GPT-4o, GPT-4o mini, o1, and o3 support multiple images per request.

**PDF processing** was significantly enhanced in 2025 with direct input support. PDFs can be up to **32 MB or 100 pages** and are processed using both text extraction and visual analysis:

```python
# Upload PDF via Files API
message_file = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="assistants"
)

# Create thread with PDF
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "Summarize this document.",
            "attachments": [
                {
                    "file_id": message_file.id,
                    "tools": [{"type": "file_search"}]
                }
            ]
        }
    ]
)

# Or use base64 encoding for direct input
with open("document.pdf", "rb") as pdf_file:
    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract key information."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:application/pdf;base64,{base64_pdf}"}
                }
            ]
        }
    ]
)
```

PDFs consume significantly more tokens as each page is processed for both text and visual content. The system handles diagrams, charts, and technical documents effectively but has limitations with image-only scans lacking text layers. Password-protected PDFs are not supported.

**Audio transcription and translation** uses the Whisper API with support for over 100 languages and files up to **25 MB** in m4a, mp3, mp4, mpeg, mpga, wav, and webm formats:

```python
# Basic transcription
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"  # Options: json, text, srt, vtt, verbose_json
)

# Translation to English
translation = client.audio.translations.create(
    model="whisper-1",
    file=open("german_audio.mp3", "rb")
)

# Detailed transcription with timestamps
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("meeting.mp3", "rb"),
    response_format="verbose_json",
    timestamp_granularities=["segment"]
)

for segment in transcript.segments:
    print(f"[{segment.start}s - {segment.end}s]: {segment.text}")
```

The verbose_json format provides segment-level timestamps, confidence scores, and detailed metadata useful for creating subtitles or analyzing speech patterns.

**Document handling through the Assistants API** supports text files, PDFs, Word documents, PowerPoint files, CSV, Markdown, and code files up to **512 MB or 5 million tokens** per file. Vector stores can contain up to 10,000 files:

```python
# Create vector store with documents
file1 = client.files.create(file=open("knowledge1.pdf", "rb"), purpose="assistants")
file2 = client.files.create(file=open("knowledge2.txt", "rb"), purpose="assistants")

vector_store = client.beta.vector_stores.create(
    name="Knowledge Base",
    file_ids=[file1.id, file2.id]
)

# Create assistant with file search
assistant = client.beta.assistants.create(
    name="Research Assistant",
    model="gpt-4o",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {
            "vector_store_ids": [vector_store.id]
        }
    }
)
```

**Video processing** lacks direct upload support but works effectively through frame extraction. GPT-4.1's 1 million token context window enables processing long videos by sampling frames:

```python
import cv2
import base64

video = cv2.VideoCapture("video.mp4")
base64_frames = []

while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()

# Sample every 30th frame for efficiency
sampled_frames = base64_frames[::30]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what happens in this video."},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                    }
                    for frame in sampled_frames
                ]
            ]
        }
    ]
)
```

The **Files API** provides general file management across different purposes. Files uploaded with `purpose="assistants"` support the Assistants API, while `purpose="batch"` enables batch processing:

```python
# Upload file
file = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="assistants"
)

# List files
files = client.files.list()

# Retrieve file info
file_info = client.files.retrieve("file-abc123")

# Get file content
file_content = client.files.content("file-abc123")

# Delete file
client.files.delete("file-abc123")
```

Best practices for multimodal processing include using URLs for long-running conversations to reduce request size, compressing images approaching the 20MB limit, ensuring PDFs have text layers rather than being image-only scans, using MP3 format for audio compression, and organizing related documents in the same vector store. For production applications processing many files, consider the Batch API for 50% cost savings.

## Installation, streaming, and function calling fundamentals

The OpenAI Python SDK is currently at **version 2.0.0** (released September 30, 2025) with **Python 3.8 or higher** required. Installation is straightforward:

```bash
pip install openai
```

Check your installed version:

```python
import openai
print(openai.__version__)
```

**The v1.0.0 release** in November 2023 introduced breaking changes requiring significant code updates. The most important change involves client instantiation. The old module-level API (`openai.api_key = "key"` and `openai.ChatCompletion.create()`) was replaced with explicit client objects:

```python
# Old v0.x approach (deprecated)
import openai
openai.api_key = "your-key"
response = openai.ChatCompletion.create(...)

# New v1.x+ approach (current)
from openai import OpenAI
client = OpenAI(api_key="your-key")  # or use OPENAI_API_KEY env var
response = client.chat.completions.create(...)
```

Method names changed across all endpoints. Chat completions moved from `openai.ChatCompletion.create()` to `client.chat.completions.create()`. Embeddings changed from `openai.Embedding.create()` to `client.embeddings.create()`. Audio transcription shifted from `openai.Audio.transcribe()` to `client.audio.transcriptions.create()`. Image generation moved from `openai.Image.create()` to `client.images.generate()`.

Response objects switched from dictionary access to **Pydantic models with dot notation**. Instead of `response['choices'][0]['message']['content']`, you now use `response.choices[0].message.content`. Error classes moved from `openai.error` to the main `openai` namespace, so `openai.error.RateLimitError` became `openai.RateLimitError`.

OpenAI provides an **automatic migration tool** called grit:

```bash
curl -fsSL https://docs.grit.io/install | bash
grit install
grit apply openai
```

Note that automatic migration is not supported for Azure OpenAI customers, who must migrate manually.

**Streaming responses** enable real-time output for better user experience:

```python
from openai import OpenAI

client = OpenAI()

# Synchronous streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a story about a unicorn."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# Asynchronous streaming
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def stream_response():
    stream = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

asyncio.run(stream_response())
```

The Responses API uses the same streaming pattern but with different event structures that include reasoning and tool execution events.

**Function calling** enables models to invoke external tools and APIs. The implementation requires defining function schemas, making an initial API call, executing called functions, and sending results back:

```python
import json

# Define function
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }
    return json.dumps(weather_info)

# Define function schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Initial API call
messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # or force with {"type": "function", "function": {"name": "..."}}
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

if tool_calls:
    messages.append(response_message)
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "get_current_weather":
            function_response = get_current_weather(
                location=function_args.get("location"),
                unit=function_args.get("unit", "fahrenheit")
            )
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response
            })
    
    second_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    print(second_response.choices[0].message.content)
```

**Error handling** requires comprehensive coverage of network issues, rate limits, and API errors:

```python
import openai
import time

def make_api_call_with_retry(max_retries=3):
    retries = 0
    
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            return response
            
        except openai.APIConnectionError as e:
            print(f"Connection error: {e}")
            retries += 1
            time.sleep(2 ** retries)  # Exponential backoff
            
        except openai.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            retry_after = getattr(e, 'retry_after', 30)
            time.sleep(retry_after)
            retries += 1
            
        except openai.AuthenticationError as e:
            print(f"Authentication failed: {e}")
            raise  # Don't retry auth errors
            
        except openai.BadRequestError as e:
            print(f"Bad request: {e.status_code}")
            raise  # Don't retry malformed requests
            
        except openai.InternalServerError as e:
            print(f"Server error: {e}")
            retries += 1
            time.sleep(2 ** retries)
    
    raise Exception(f"Max retries exceeded")
```

The SDK implements **automatic retries** for connection errors, 408 Request Timeout, 409 Conflict, 429 Rate Limit, and 500+ server errors. Configure retry behavior globally or per-request:

```python
# Global retry configuration
client = OpenAI(max_retries=5)

# Per-request override
client.with_options(max_retries=3).chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Advanced configuration includes timeout control, proxy support, and Azure integration:

```python
import httpx

# Custom timeouts
client = OpenAI(
    timeout=httpx.Timeout(
        connect=5.0,
        read=10.0,
        write=10.0,
        pool=5.0
    )
)

# Proxy configuration
client = OpenAI(
    http_client=DefaultHttpxClient(
        proxy="http://my.proxy.com:8080"
    )
)

# Azure OpenAI
from openai import AzureOpenAI

azure_client = AzureOpenAI(
    api_version="2023-07-01-preview",
    azure_endpoint="https://example-endpoint.openai.azure.com",
    api_key="your-azure-api-key"
)
```

## Embeddings, batch processing, and rate limits for production

**Embeddings models** convert text into vector representations for semantic search, clustering, and retrieval-augmented generation. OpenAI offers three models with different performance-cost tradeoffs. The **text-embedding-3-large** model provides the highest accuracy with up to 3,072 dimensions, achieving 54.9% on MIRACL (vs 31.4% for ada-002) and 64.6% on MTEB (vs 61.0% for ada-002), priced at $0.13/1M tokens. The **text-embedding-3-small** model offers excellent cost-efficiency at 1,536 dimensions with 44.0% MIRACL and 62.3% MTEB scores for $0.02/1M tokens, representing 5x cost savings over ada-002. The legacy **text-embedding-ada-002** at $0.10/1M tokens remains supported but newer models are recommended.

Both text-embedding-3 models support **dimension reduction** through Matryoshka Representation Learning, allowing you to shorten embeddings by up to 6x without significant quality loss:

```python
from openai import OpenAI

client = OpenAI()

# Standard embeddings
response = client.embeddings.create(
    input=["Text to embed", "Another document"],
    model="text-embedding-3-small"
)

embeddings = [data.embedding for data in response.data]

# Custom dimensions for storage optimization
response = client.embeddings.create(
    input="Your text here",
    model="text-embedding-3-large",
    dimensions=512  # Reduces from 3072 to 512
)

# Async usage for high throughput
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def get_embeddings():
    response = await async_client.embeddings.create(
        input=["Document 1", "Document 2", "Document 3"],
        model="text-embedding-3-small"
    )
    return response.data

embeddings = await get_embeddings()
```

Use text-embedding-3-large when accuracy is critical for complex semantic tasks. Use text-embedding-3-small for general-purpose applications where cost efficiency matters. The dimension reduction feature enables text-embedding-3-large at 256 dimensions to outperform ada-002 at full 1,536 dimensions while using less storage.

**The Batch API** provides asynchronous processing with **50% cost reduction** compared to standard API calls. Jobs complete within a 24-hour window, often much faster, with separate quota that doesn't disrupt online workloads. Supported endpoints include chat completions and embeddings:

```python
import json
import time

# Step 1: Prepare batch requests in JSONL format
batch_requests = [
    {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 100
        }
    },
    {
        "custom_id": "request-2",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "max_tokens": 100
        }
    }
]

# Save to JSONL file
with open("batch_requests.jsonl", "w") as f:
    for request in batch_requests:
        f.write(json.dumps(request) + "\n")

# Step 2: Upload file
batch_input_file = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch"
)

# Step 3: Create batch job
batch = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Step 4: Poll for completion
while True:
    batch_status = client.batches.retrieve(batch.id)
    print(f"Status: {batch_status.status}")
    
    if batch_status.status == "completed":
        break
    elif batch_status.status in ["failed", "expired", "cancelled"]:
        print(f"Batch {batch_status.status}")
        break
    
    time.sleep(60)

# Step 5: Download results
if batch_status.status == "completed":
    result_file_id = batch_status.output_file_id
    results = client.files.content(result_file_id)
    
    for line in results.text.strip().split('\n'):
        result = json.loads(line)
        print(f"Request {result['custom_id']}: {result['response']['body']}")
```

Batch processing works excellently for embeddings generation, evaluation datasets, content classification, and any non-urgent workloads. The 50% cost savings make it ideal for large-scale data processing.

**Rate limits** use a tiered system based on spending history and are enforced at the organization level. Limits are measured in four metrics: RPM (requests per minute), TPM (tokens per minute), RPD (requests per day), and TPD (tokens per day). OpenAI automatically upgrades your tier based on usage: Free tier ($0 spent) has minimal limits like 3 RPM for GPT-4, Tier 1 ($5+ paid) provides around 500K TPM for GPT-4o with 1K RPM, and higher tiers (2-5) unlock progressively higher limits based on spend and time thresholds.

Recent improvements include GPT-4o Tier 1+ users receiving **500K TPM** compared to previous limits. Embeddings models have generous limits of 5,000 RPM and 5,000,000 TPM across all tiers. The Batch API operates on separate quota, allowing high-volume processing without affecting real-time workloads.

Implement exponential backoff for rate limit handling:

```python
import openai
import time

def call_with_retry(func, max_retries=5):
    """Retry with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
            print(f"Rate limit hit. Waiting {wait_time}s...")
            time.sleep(wait_time)

result = call_with_retry(
    lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )
)
```

Monitor your specific rate limits at `https://platform.openai.com/settings/organization/limits` as exact limits are no longer published in documentation but visible in your account dashboard.

**Production best practices** include storing API keys in environment variables, implementing comprehensive error handling with retry logic, setting appropriate timeouts for your use case, monitoring rate limit headers, using streaming for better user experience, leveraging async clients for concurrent operations, and testing thoroughly after version upgrades.

Cost optimization strategies include using the Batch API for non-urgent tasks (50% savings), selecting appropriate models (gpt-4o-mini vs gpt-4o), optimizing prompts to reduce token usage, caching responses when possible, and implementing prompt compression for long contexts. Token counting with tiktoken helps predict costs:

```python
import tiktoken

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

text = "Your prompt here"
token_count = count_tokens(text)

if token_count > 8000:
    print(f"Warning: {token_count} tokens may be expensive")
```

For production deployments, implement comprehensive logging, set spending limits in your OpenAI account dashboard, validate inputs before API calls, implement content filtering using the moderation endpoint, and build fallback strategies:

```python
def chat_with_fallback(messages):
    """Try primary model, fallback to cheaper alternatives"""
    models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    
    for model in models:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages
            )
        except openai.RateLimitError:
            continue
    
    raise Exception("All models rate limited")
```

The combination of embeddings for semantic search, batch processing for cost efficiency, and proper rate limit handling creates robust production systems capable of handling enterprise-scale workloads while optimizing costs and reliability.
