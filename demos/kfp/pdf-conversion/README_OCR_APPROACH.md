# Docling OCR-Only Pipeline vs VLM Pipeline

This directory contains two different approaches for processing images in a RAG system:

## 1. VLM-Based Pipeline (`docling_convert_images_pipeline_with_vlm.py`)

**Architecture**: Two-stage processing
- **Stage 1**: Vision Language Model (LLaVA) analyzes images for visual content understanding
- **Stage 2**: Text-based LLM for RAG-based question answering

**Capabilities**:
- ✅ Visual content understanding (objects, colors, scenes, artistic analysis)
- ✅ Text extraction from images
- ✅ Complex visual reasoning
- ✅ Comprehensive image analysis

**Requirements**:
- GPU with compute capability 8.0+ for quantized models
- VLM model deployment (LLaVA or similar)
- Higher resource requirements

**Use Cases**:
- Images with complex visual content
- Artistic analysis
- Scene understanding
- When you need to understand "what's in the image" beyond just text

## 2. OCR-Only Pipeline (`docling_convert_images_pipeline_ocr_only.py`)

**Architecture**: Single-stage OCR processing
- **Stage 1**: Docling OCR extracts text from images
- **Stage 2**: Text-based LLM for RAG-based question answering

**Capabilities**:
- ✅ Text extraction from images (OCR)
- ✅ Technical document processing
- ✅ Lower resource requirements
- ✅ No GPU dependency
- ❌ Limited to text-rich images
- ❌ No visual content understanding

**Requirements**:
- CPU-only processing possible
- No VLM model needed
- Lower resource requirements

**Use Cases**:
- Documents with text (PDFs converted to images, scanned documents)
- Screenshots of text
- Technical diagrams with labels
- Any image where the main content is readable text

## Key Differences

| Feature | VLM Pipeline | OCR-Only Pipeline |
|---------|-------------|------------------|
| GPU Requirement | Yes (8.0+ compute capability) | No |
| Resource Usage | High | Low |
| Text Extraction | ✅ | ✅ |
| Visual Understanding | ✅ | ❌ |
| Scene Analysis | ✅ | ❌ |
| Object Recognition | ✅ | ❌ |
| Processing Speed | Slower | Faster |
| Deployment Complexity | Higher | Lower |

## When to Use Which Approach

### Use OCR-Only Pipeline When:
- Your images primarily contain text content
- You're processing technical documents, screenshots, or scanned documents
- You want lower resource requirements and faster processing
- You don't need visual content understanding
- You're having GPU compatibility issues

### Use VLM Pipeline When:
- Your images contain complex visual content
- You need to understand objects, scenes, or artistic elements
- You want comprehensive image analysis beyond just text
- You have compatible GPU hardware available
- Processing speed is less critical than analysis depth

## Example Image Types

### Good for OCR-Only:
- Scanned documents
- Screenshots of code or text
- Technical diagrams with labels
- Presentations converted to images
- Forms and tables

### Requires VLM:
- Photographs of people, places, or objects
- Artistic images (paintings, drawings)
- Complex scenes requiring interpretation
- Images where understanding context matters more than reading text
- Medical images, satellite imagery, etc.

## Running the OCR-Only Pipeline

```bash
# Compile the pipeline
python docling_convert_images_pipeline_ocr_only.py

# The compiled YAML will be: docling_convert_pipeline_ocr_only.yaml
```

## Configuration

The OCR-only pipeline uses the same LlamaStack service and Milvus vector database but:
- Removes VLM service dependency
- Uses Docling's built-in OCR capabilities
- Processes images directly with CPU
- Creates embeddings from extracted text only

This makes it much simpler to deploy and more reliable for text-heavy image processing tasks. 