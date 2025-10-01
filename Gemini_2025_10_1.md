# Google Gemini Python SDK: Complete Technical Guide

Google's new **google-genai** Python SDK (v1.0+, GA since May 2025) provides production-ready access to Gemini models with powerful capabilities for structured JSON output, multimodal file processing, and advanced reasoning. The SDK replaces the deprecated google-generativeai library and offers unified access to both Gemini API and Vertex AI. **Most critically: it enables guaranteed JSON schema compliance through response_schema and supports uploading videos up to 2GB with automatic processing**—making it ideal for building reliable data extraction pipelines and multimodal AI applications.

## Installation and setup in under 2 minutes

The setup process requires Python 3.9+ and takes just three commands. Install the SDK with `pip install -U google-genai`, obtain your API key from aistudio.google.com/app/apikey, and export it as an environment variable with `export GEMINI_API_KEY="your-key-here"`. The client automatically detects this environment variable, enabling immediate usage without hardcoding credentials.

```python
from google import genai
from google.genai import types

# Client auto-detects GEMINI_API_KEY from environment
client = genai.Client()

# Basic text generation
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Explain machine learning in simple terms'
)
print(response.text)
```

The SDK supports **gemini-2.5-flash** (best price-performance for high volume), **gemini-2.5-pro** (advanced reasoning for complex tasks), and earlier 2.0/1.5 model versions. For Vertex AI deployment on Google Cloud, initialize with `client = genai.Client(vertexai=True, project='your-project-id', location='us-central1')` instead. This unified client interface works seamlessly across both environments.

## JSON schema responses: Guaranteed structured output

The combination of `response_mime_type` and `response_schema` parameters provides Gemini's most powerful feature for production applications—**deterministic JSON output that strictly adheres to predefined schemas**. This eliminates unreliable prompt-based JSON generation and enables reliable downstream processing for databases, APIs, and data pipelines.

### Core mechanism with Pydantic models

The recommended approach uses Pydantic models to define schemas. When configured, the model is **constrained** to produce only valid JSON matching your schema—not merely encouraged. The SDK's `.parsed` attribute returns type-safe Python objects automatically.

```python
from google import genai
from pydantic import BaseModel

class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]
    prep_time_minutes: int

client = genai.Client(api_key='YOUR_API_KEY')

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='List 3 popular cookie recipes with prep times',
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Recipe],
    }
)

# Type-safe access to parsed objects
my_recipes: list[Recipe] = response.parsed
for recipe in my_recipes:
    print(f"{recipe.recipe_name}: {recipe.prep_time_minutes} mins")
    
# Or raw JSON string
print(response.text)
```

This pattern supports complex nested structures with full Python typing. The **response_mime_type** parameter must be set to `application/json` to enable JSON mode, while **response_schema** defines the structure. Supported schemas include Pydantic models (recommended), TypedDict, Python type annotations like `list[dict[str, int]]`, or OpenAPI 3.0 schema dictionaries.

### Advanced schema patterns for real applications

Complex business logic often requires enums, optional fields, and nested objects. Pydantic Field validators enable precise control over data validation including minimum/maximum values, string patterns, and field descriptions that guide the model's output.

```python
from pydantic import BaseModel, Field
from typing import Optional
import enum

class Priority(enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Task(BaseModel):
    title: str = Field(description="Task title")
    priority: Priority
    estimated_hours: int = Field(ge=1, le=40, description="Hours estimate")
    assignee: Optional[str] = None

class ProjectAnalysis(BaseModel):
    project_name: str
    tasks: list[Task]
    total_hours: int

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Analyze this project scope: Build a REST API with authentication, user management, and analytics dashboard',
    config={
        'response_mime_type': 'application/json',
        'response_schema': ProjectAnalysis
    }
)

analysis: ProjectAnalysis = response.parsed
print(f"Project: {analysis.project_name}")
print(f"Total estimated hours: {analysis.total_hours}")
for task in analysis.tasks:
    print(f"- {task.title} ({task.priority.value}): {task.estimated_hours}h")
```

The SDK automatically handles **property ordering** (alphabetically by default) but you can specify custom ordering with the `propertyOrdering` field in raw schema definitions. For enum classification tasks where you need a single enum value rather than a JSON object, use `response_mime_type='text/x.enum'` instead.

### Schema limitations and troubleshooting

While extremely powerful, JSON schema has practical constraints. Schema complexity contributes to the input token limit—large schemas reduce available tokens for prompts and responses. Complex schemas with many optional properties, long array length limits, extensive enums, or multiple constraint types can trigger **InvalidArgument 400 errors** even with valid syntax.

To resolve complexity errors: shorten property and enum names, reduce nested depth, simplify validation constraints (remove unnecessary min/max/format rules), and make fewer fields required. Fields are optional by default unless listed in a `required` array. Additionally, you **cannot combine** `response_mime_type: 'application/json'` with function calling tools—attempting both simultaneously returns an unsupported error.

Token usage monitoring reveals the true cost. Access `response.usage_metadata.prompt_token_count` and `response.usage_metadata.candidates_token_count` to see how schema size impacts your quota. Schemas count toward prompt tokens, so minimize unnecessary complexity.

### Best practices for production systems

Start with simple schemas and add complexity incrementally—test each addition before expanding further. Use Pydantic models rather than raw dictionary schemas for type safety and validation. Configure schemas on the model rather than describing them in prompts, which is unreliable and not enforced. Provide clear context in prompts about what data to extract and why, as the model performs better with specific instructions.

```python
# ✅ GOOD: Clear task with schema
contents = """
Extract contact information from this email:
"Hi, I'm Sarah Johnson from Acme Corp (sarah.j@acme.com, +1-555-0123). 
Please reach out if you need anything!"
"""

class Contact(BaseModel):
    name: str
    company: str
    email: str
    phone: str

# ❌ AVOID: Vague prompt without context
contents = "Process this text"
```

For data extraction workflows, consider making an initial pass asking the model to describe what it sees, then use that description in a second request for structured extraction. This two-phase approach improves accuracy for complex documents. Handle errors gracefully with try-except blocks checking for APIError with specific error codes, especially 400 for schema issues.

## Uploading and processing files and videos

The File API enables uploading media files (videos, images, audio, PDFs) up to **2GB per file** with **20GB total storage per project**. Files remain available for 48 hours with automatic deletion, though you can delete manually. This API is essential for requests exceeding 20MB or when reusing the same file across multiple requests.

### Complete video upload workflow

Video support includes mp4, mpeg, mov, avi, webm, and other common formats. **Videos require processing time** before use—always check the file state and wait for completion. Duration limits depend on context window: 1M context models support up to 1 hour (3 hours at low resolution) while 2M context models handle up to 2 hours (6 hours low resolution).

```python
from google import genai
import time

client = genai.Client(api_key='YOUR_API_KEY')

# 1. Upload video file
print("Uploading video...")
video_file = client.files.upload(file='presentation.mp4')
print(f"Uploaded: {video_file.uri}")

# 2. Wait for processing to complete
while video_file.state.name == "PROCESSING":
    print('Processing video...')
    time.sleep(10)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError(f"Video processing failed: {video_file.state}")

print("Video ready for analysis")

# 3. Generate content using the video
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        video_file,
        'Create a detailed summary with timestamps (MM:SS format) of key moments in this presentation'
    ]
)

print(response.text)

# 4. Reuse the same file for multiple analyses
quiz_response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[video_file, 'Generate 5 quiz questions based on this content']
)
```

Videos are sampled at **1 frame per second** by default with audio processed at 1Kbps. This generates approximately **300 tokens per second** of video (258 tokens/frame + 32 tokens/second audio). You can customize frame rate with the `video_metadata` parameter, setting lower FPS for static content or higher for action sequences.

### Advanced video processing capabilities

YouTube URLs work directly without uploading, though with limitations: free tier supports 8 hours daily while paid has no duration limits. Only public or unlisted videos are accessible, with a maximum of 10 videos per request on Gemini 2.5+ models.

```python
from google.genai import types

# YouTube URL processing
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(
                    file_uri='https://www.youtube.com/watch?v=VIDEO_ID'
                ),
                video_metadata=types.VideoMetadata(
                    start_offset='125s',  # Start at 2:05
                    end_offset='245s'      # End at 4:05
                )
            ),
            types.Part(text='Analyze this video segment')
        ]
    )
)

# Custom frame rate for detailed analysis
video_bytes = open('detailed_demo.mp4', 'rb').read()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=types.Content(
        parts=[
            types.Part(
                inline_data=types.Blob(
                    data=video_bytes,
                    mime_type='video/mp4'
                ),
                video_metadata=types.VideoMetadata(fps=5)  # 5 frames/second
            ),
            types.Part(text='Describe the visual changes in detail')
        ]
    )
)
```

For videos under 20MB, inline upload is possible using `inline_data` with `types.Blob`, but the File API approach is recommended for consistency and reusability. Reference specific timestamps in prompts using MM:SS format for precise temporal queries like "What happens at 02:30?"

### Image, document, and audio upload patterns

Images support png, jpeg, webp, heic, and heif formats. The SDK accepts PIL Image objects directly, file paths, URIs (including Google Cloud Storage gs:// URLs), or raw bytes. Images are automatically scaled to a maximum of **3072x3072 pixels** while preserving aspect ratio, and you can process up to **3,000 images per batch request**.

```python
from PIL import Image

# Option 1: Upload via File API
image_file = client.files.upload(file='chart.png')

# Option 2: Direct PIL Image object
image = Image.open('chart.png')

# Option 3: From URI
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        'Analyze this chart and extract the data',
        types.Part.from_uri(
            file_uri='gs://my-bucket/chart.png',
            mime_type='image/png'
        )
    ]
)

# Option 4: From bytes
with open('chart.png', 'rb') as f:
    image_bytes = f.read()
    
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        'What trends are shown?',
        types.Part.from_bytes(data=image_bytes, mime_type='image/png')
    ]
)
```

PDF documents are the primary document format, with a **30MB recommended maximum** and up to **1,000 pages**. Each page consumes 258 tokens as PDFs are treated as images (one page equals one image). Text formats like txt, html, css, markdown, csv, xml, and rtf are also supported but processed as pure text extraction.

```python
import io
import httpx

# Upload large PDF from URL
pdf_url = "https://example.com/whitepaper.pdf"
pdf_data = httpx.get(pdf_url).content
pdf_io = io.BytesIO(pdf_data)

pdf_file = client.files.upload(
    file=pdf_io,
    config={'mime_type': 'application/pdf'}
)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        pdf_file,
        'Extract all the key findings and statistics from this research paper'
    ]
)
```

Audio files in wav, mp3, aiff, aac, ogg, and flac formats upload similarly. The Gemini Apps free tier supports 10 minutes of audio while Pro/Ultra tiers allow 3 hours. All files uploaded via the File API can be listed with `client.files.list()`, retrieved with `client.files.get(name=file.name)`, and deleted with `client.files.delete(name=file.name)`.

### File management and multimodal combinations

The reusability of uploaded files provides significant efficiency for multiple analyses. Upload once and reference the file object in unlimited requests within the 48-hour window. Combine multiple file types in a single request for cross-modal analysis.

```python
# Multi-modal analysis example
video = client.files.upload(file='product_demo.mp4')
pdf = client.files.upload(file='product_specs.pdf')
image = client.files.upload(file='competitor_chart.png')

# Wait for video processing
while video.state.name == "PROCESSING":
    time.sleep(5)
    video = client.files.get(name=video.name)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        'Compare our product demo with the specifications document and competitive analysis chart. Identify gaps and opportunities.',
        video,
        pdf,
        image
    ]
)

# List all uploaded files
print('Uploaded files:')
for file in client.files.list():
    print(f'  {file.display_name or file.name} - {file.size_bytes} bytes - {file.state.name}')

# Cleanup when done (optional - auto-deletes after 48h)
client.files.delete(name=video.name)
client.files.delete(name=pdf.name)
client.files.delete(name=image.name)
```

Best practices for file uploads include always using the File API for files exceeding 20MB, waiting for video processing state to reach "ACTIVE" before generating content, and placing files before text prompts for better results. For long videos exceeding 1-2 hours, split into segments and process separately. Use error handling with try-except blocks to catch upload failures and processing errors.

## Thinking mode for complex reasoning tasks

Gemini 2.5 models include a native **thinking mode** capability where the model generates internal reasoning steps before producing final answers. This significantly improves performance on complex tasks requiring multi-step logic like advanced mathematics, algorithm design, and intricate problem solving. All Gemini 2.5 models (Pro, Flash, Flash-Lite) support thinking with **dynamic thinking enabled by default** on Pro and Flash.

The `ThinkingConfig` class controls thinking behavior through two parameters. The **thinking_budget** parameter sets token allocation for internal reasoning: `-1` enables dynamic thinking where the model decides based on task complexity (default), `0` disables thinking entirely (Flash and Flash-Lite only, not available for Pro), and specific values from 128-32,768 (Pro) or 0-24,576 (Flash/Flash-Lite) set explicit budgets.

```python
from google import genai
from google.genai import types

client = genai.Client()

# High thinking budget for complex problems
response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents='Solve: Find the sum of all integer bases b > 9 for which 17_b is a divisor of 97_b',
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=10000)
    )
)
print(response.text)

# Disable thinking for simple queries (Flash only)
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What is the capital of France?',
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
)

# View thought summaries for debugging
response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents='Complex logic puzzle...',
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    )
)

for part in response.candidates[0].content.parts:
    if part.thought:
        print("Thought summary:", part.text)
    else:
        print("Answer:", part.text)
```

The **include_thoughts** parameter (default False) controls whether thought summaries appear in responses. When enabled, you can inspect the model's reasoning process for debugging unexpected results. Note that while the model generates full "raw thoughts" internally, only condensed summaries are returned—but you're billed for the complete thinking token count accessible via `response.usage_metadata.thoughts_token_count`.

Task complexity determines optimal settings: disable thinking for simple fact retrieval to save costs, use dynamic thinking for moderate analysis tasks, and set high budgets (10,000+ tokens) for advanced mathematics or complex coding challenges. Thinking mode integrates seamlessly with all Gemini tools including code execution, function calling, and Google Search, enabling sophisticated multi-step workflows.

## Additional SDK capabilities and configuration

Beyond the core features, the SDK supports streaming responses with `generate_content_stream()`, multi-turn chat sessions via `client.chats.create()`, system instructions for defining behavior, and safety settings for content filtering. Function calling enables automatic tool use by passing Python functions with docstrings as tools.

```python
# Streaming for progressive output
for chunk in client.models.generate_content_stream(
    model='gemini-2.5-flash',
    contents='Write a detailed guide to machine learning'
):
    print(chunk.text, end='', flush=True)

# Chat sessions with context
chat = client.chats.create(model='gemini-2.5-flash')
response1 = chat.send_message('My name is Alex')
response2 = chat.send_message('What did I just tell you?')  # Remembers context

# System instructions and configuration
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Explain photosynthesis',
    config=types.GenerateContentConfig(
        system_instruction='You are a biology teacher. Use simple language and analogies.',
        temperature=0.7,
        max_output_tokens=500,
        top_p=0.95
    )
)

# Token counting for cost estimation
token_count = client.models.count_tokens(
    model='gemini-2.5-flash',
    contents='Your prompt here'
)
print(f"This prompt uses {token_count.total_tokens} tokens")
```

The SDK includes async support via `client.aio` for non-blocking operations in async applications. Vertex AI users access identical functionality through the unified client by setting `vertexai=True` during initialization. Rate limits and quota vary by model and tier, with the SDK automatically handling retries for transient errors.

Migration from the legacy google-generativeai package requires updating imports from `import google.generativeai as genai` to `from google import genai` and changing the initialization pattern from `genai.configure()` and `GenerativeModel` to the new `Client` approach. The new SDK provides superior type safety, direct access to parsed Pydantic objects via `.parsed`, and unified Vertex AI integration.

## Key implementation patterns and recommendations

For production applications, combine JSON schema responses with file uploads to build robust data extraction pipelines. Upload documents or videos, extract structured information matching your schema, and pipe results directly to databases or downstream services. This eliminates brittle text parsing and provides guaranteed output formats.

Set appropriate thinking budgets based on task complexity—use dynamic thinking as the default, but disable for simple queries to optimize latency and costs. Monitor token usage through `usage_metadata` to understand actual consumption including thinking tokens. Always handle video processing states properly by polling until "ACTIVE" before generation requests.

Structure prompts clearly with specific instructions about desired output format and content. For JSON extraction, describe the data to extract and why each field matters. For video analysis, specify whether you need timestamps, transcriptions, visual descriptions, or combinations. Place uploaded files before text prompts in the contents array for optimal results.

Error handling should catch `APIError` exceptions with specific status code checks for 400 (invalid schema/request), 404 (file not found), and rate limit errors. Implement exponential backoff for retries and graceful degradation when possible. Use environment variables for API keys in production rather than hardcoding credentials.

The Gemini Python SDK represents a production-grade solution for building AI applications with guaranteed structured output, multimodal understanding, and advanced reasoning capabilities, all through a clean, Pythonic interface that works identically across Gemini API and Vertex AI deployments.
