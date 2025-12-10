# Bible Verse Reference Extractor

A fine-tuned T5 model that extracts structured Bible verse references from natural language text.

## Overview

This project trains a T5-small model to identify and extract Bible verse references from conversational text and return them in a structured JSON format. The model can handle various reference formats including standard notation (John 3:16), verbose formats (John Chapter 3 verse 16), abbreviated forms, and chapter-only references.

## Model Details

- **Base Model**: T5-small (60M parameters)
- **Task**: Sequence-to-sequence text generation
- **Training Data**: 5,000 synthetically generated examples
- **Output Format**: JSON with Book, Chapter, and Verse fields

## Supported Reference Formats

The model recognizes multiple Bible reference formats:

- **Standard**: "John 3:16"
- **Verbose**: "John Chapter 3 verse 16"
- **Abbreviated**: "1 Cor 3 v 16"
- **Chapter Only**: "Psalm 23"
- **Embedded in Context**: "Let's open our bibles to Matthew 5:3 today."

## Installation

```bash
pip install transformers torch sentencepiece
```

## Usage

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

# Load model and tokenizer
model_name = "ThatLinuxGuyYouKnow/bible_extractor_v1"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_reference(text):
    # Prepare input with instruction prompt
    input_text = f"extract bible reference: {text}"
    
    # Tokenize and generate
    inputs = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=128)
    
    # Decode output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse JSON
    return json.loads(result)

# Example
text = "I love the verse John 3:16 for inspiration."
reference = extract_reference(text)
print(reference)
# Output: {"Book": "John", "Chapter": "3", "Verse": "16"}
```

### Batch Processing

```python
test_sentences = [
    "Open your bibles to Genesis 1:1 right now.",
    "The pastor was reading from 1 Kings Chapter 12 verse 4",
    "Check out Rev 20:10"
]

for sentence in test_sentences:
    try:
        reference = extract_reference(sentence)
        print(f"Input: {sentence}")
        print(f"Extracted: {reference}\n")
    except json.JSONDecodeError:
        print(f"Could not parse: {sentence}\n")
```

## Output Format

The model returns a JSON object with the following structure:

```json
{
  "Book": "Book name",
  "Chapter": "Chapter number",
  "Verse": "Verse number or null"
}
```

Note: The `Verse` field will be `null` for chapter-only references (e.g., "Psalm 23").

## Training Details

### Dataset

- **Size**: 5,000 synthetic examples (4,500 train, 500 test)
- **Generation**: Programmatically created with random book/chapter/verse combinations
- **Variations**: Multiple reference formats and contextual wrappings

### Hyperparameters

- Learning Rate: 3e-4
- Batch Size: 16
- Epochs: 10
- Max Input Length: 128 tokens
- Max Output Length: 128 tokens
- Mixed Precision: FP16 enabled

### Covered Books

The model is trained on all 66 canonical Bible books, from Genesis to Revelation.

## Limitations

- Trained exclusively on synthetic data - may not generalize to all real-world variations
- Performance may vary with highly unconventional reference formats
- Does not validate whether the referenced verse actually exists in the Bible
- May struggle with non-English references or heavily abbreviated book names
- Best suited for English text

## Future Improvements

- Train on real sermon transcripts and Bible study materials
- Add support for verse ranges (e.g., "John 3:16-18")
- Include multiple verse references in a single text
- Support for different Bible translations
- Verse existence validation
- Multi-language support

## Training the Model

To retrain or fine-tune the model yourself:

1. Clone this repository
2. Install dependencies: `pip install transformers datasets sentencepiece accelerate evaluate`
3. Run the training script: `python verse_extractor_training.py`
4. The model will be saved to `./bible_t5_model_v1/`

The training script generates synthetic data on-the-fly, so you can easily modify the data generation logic to add more variations or formats.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas where contributions would be especially valuable:

- Adding support for verse ranges
- Improving handling of abbreviated book names
- Training data from real-world sources
- Multi-language support


