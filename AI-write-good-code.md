## The "Look How Smart I Am" Problem

AI models are trained on vast amounts of code from GitHub, Stack Overflow, and technical blogs. This means they've absorbed patterns like:

- **Fancy design patterns** (Strategy, Factory, Observer) even when a simple function would do
- **Premature abstractions** - creating interfaces and base classes "for future flexibility" that never comes
- **Clever one-liners** that compress 10 lines into 1 unreadable line
- **Buzzword-driven architecture** - microservices, dependency injection, middleware layers for a 200-line script

## What AI *thinks* good code is:
```python
class DataProcessorFactory:
    @staticmethod
    def create_processor(processor_type: str) -> IDataProcessor:
        if processor_type == "csv":
            return CSVDataProcessor()
        elif processor_type == "json":
            return JSONDataProcessor()
```

## What a junior dev actually needs:
```python
def process_csv(filename):
    # Read the CSV and return the data
    with open(filename) as f:
        return list(csv.reader(f))
```

## Why this happens:

1. **Training data bias** - AI learns from code written by senior devs showing off, not from the boring, simple code that actually ships products
2. **No context about team** - AI doesn't know you have 3 junior devs who need to maintain this at 2am
3. **Optimization for impressiveness** - The code "looks professional" with all the patterns and abstractions
4. **Missing the "why"** - Good abstraction comes from *actual* repeated pain points, not predicted future needs

## The real gift:

**Simple, boring, obvious code** that:
- A junior can debug without Googling design patterns
- Has clear variable names like `user_email` not `ue` or `subject`
- Does one thing per function
- Has comments explaining *why*, not *what*
- Uses standard library over clever imports
- Prefers explicit over implicit
- Values readability over cleverness

The best code reads like a children's book: simple sentences, clear progression, no surprises.
