# PDF Data Extractor for Investment Analysis

A tool for extracting key investment information from earnings call transcripts and financial reports, designed for investors looking to evaluate companies.

## Overview

This project provides a comprehensive solution for extracting and analyzing key investment information from PDF documents, particularly earnings call transcripts. It uses natural language processing and pattern matching techniques to identify:

- Growth prospects and drivers
- Business changes and strategic initiatives
- Key triggers and catalysts
- Material factors affecting future earnings and growth
- Financial metrics and guidance
- Risks and challenges

## Features

- **High-Quality PDF Text Extraction**: Uses pdfplumber for superior text extraction from PDF files
- **Financial Metrics Extraction**: Identifies revenue figures, growth rates, margins, etc.
- **Investment Insight Classification**: Categorizes insights by impact level and timeframe
- **Acquisition Analysis**: Special focus on analyzing acquisition details and impact
- **Executive Summary Generation**: Creates concise summaries using OpenAI's GPT (optional)
- **Comprehensive Report Generation**: Produces a well-structured investment report
- **Customizable Analysis**: Can be tailored for specific industries or companies

## Requirements

- Python 3.7+
- Required Python packages (see `requirements.txt`):
  - openai>=0.27.0 (for GPT-based summary generation)
  - pdfplumber>=0.7.0 (for high-quality PDF text extraction)
  - nltk>=3.7 (for natural language processing)
  - textstat>=0.7.3 (for readability analysis)
  - pandas>=1.3.5 (for data manipulation)
  - numpy>=1.22.0 (numerical operations)
  - matplotlib>=3.5.0 (plotting - pre-installed in Colab)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pdf-data-extractor.git
cd pdf-data-extractor
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up your OpenAI API key if you want to use GPT for summary generation:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## Usage

### Basic Usage

```python
from pdf_extractor import PDFInvestmentAnalyzer

# Initialize the analyzer
analyzer = PDFInvestmentAnalyzer()

# Extract text from a PDF file
text = analyzer.extract_text_from_pdf("path/to/earnings_call.pdf")

# Analyze the document
results = analyzer.analyze_document(text, company_name="Company Name")

# Generate and print the report
report = analyzer.generate_investment_report(results)
print(report)

# Save results to JSON for further analysis
import json
with open('investment_analysis.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
```

### Using with Google Colab

This tool is designed to work seamlessly with Google Colab. Here's how to use it:

**🔗 Quick Start: [Open in Google Colab](https://colab.research.google.com/github/petrosbeyene/pdf-data-extractor/blob/main/pdf_extractor_and_analyzer.ipynb)**

1. Upload the script to your Google Colab notebook or copy the code from `pdf_extractor.py`
2. Mount your Google Drive to access PDF files:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Install required packages:
```python
!pip install openai pdfplumber nltk textstat pandas
# Note: matplotlib, numpy are pre-installed in Colab
```

4. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')  # Important for newer NLTK versions
nltk.download('stopwords')
nltk.download('wordnet')
```

5. Run the analysis:
```python
# Initialize analyzer
analyzer = PDFInvestmentAnalyzer()

# Set your PDF path (adjust to your Google Drive structure)
pdf_path = "/content/drive/MyDrive/PDFs/your_earnings_call.pdf"

# Run analysis
text_content = analyzer.extract_text_from_pdf(pdf_path)
results = analyzer.analyze_document(text_content, company_name="Your Company")
report = analyzer.generate_investment_report(results)
print(report)
```

**Note**: For Google Drive PDFs, you need to upload the PDF to your Drive first, then use the mounted path (not direct Drive links).

## Customization

### Customizing Keywords

You can customize the keywords used for analysis by modifying the initialization parameters:

```python
analyzer = PDFInvestmentAnalyzer()

# Add industry-specific growth keywords
analyzer.growth_keywords.extend(['market share', 'penetration', 'adoption'])

# Add company-specific segments
analyzer.company_segments['new_segment'] = ['keyword1', 'keyword2']
```

### Using OpenAI GPT for Enhanced Analysis

For more sophisticated analysis, you can provide an OpenAI API key:

```python
analyzer = PDFInvestmentAnalyzer(openai_api_key="your-openai-key")
```

This will enable the GPT-powered executive summary generation.

## Output Format

The analysis results are structured as follows:

- **executive_summary**: A concise summary of key investment points
- **financial_metrics**: Extracted financial figures and percentages
- **growth_insights**: List of growth-related insights with impact levels
- **business_changes**: Strategic initiatives and business transformations
- **financial_guidance**: Forward-looking statements and guidance
- **risks_challenges**: Identified risks and challenges
- **acquisition_analysis**: Detailed analysis of acquisitions mentioned
- **analysis_metadata**: Information about the analysis process

## Example Report

The generated report includes sections like:

- **📊 Executive Summary**: AI-powered summary of key investment points
- **💰 Key Financial Metrics**: Revenue figures, growth rates, margins with context
- **🔄 Acquisition Analysis**: Details, financial impact, strategic benefits
- **🚀 Growth Drivers & Prospects**: Growth opportunities with impact assessment
- **🔄 Key Business Changes**: Strategic initiatives and transformations
- **🎯 Financial Guidance & Outlook**: Forward-looking statements and targets
- **⚠️ Risks & Challenges**: Identified risks with impact levels
- **💡 Investment Thesis**: Key investment points and recommendations
- **📈 Analysis Metadata**: Processing statistics and analysis quality metrics

### Sample Output
```
================================================================================
SJS ENTERPRISES - INVESTMENT ANALYSIS REPORT
================================================================================

💰 KEY FINANCIAL METRICS
--------------------------------------------------
Growth Rates: 13.6, 21, 24.6, 18.6
Margin Figures: 31.5, 27.3, 12.8

🚀 GROWTH DRIVERS & PROSPECTS
--------------------------------------------------
• Walter Pack Q1 witnessed strong revenue growth of 21% YoY and robust margin performance with EBITDA margins around 31.5%
  Evidence: 21, 31.5
  Impact: High | Timeframe: Medium-term
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Files in This Repository

- `pdf_extractor.py` - Main Python script with all analysis functionality
- `pdf_extractor.ipynb` - Google Colab notebook version
- `requirements.txt` - Python package dependencies
- `README.md` - This documentation

## Demo

Try the live demo: [Google Colab Notebook](https://colab.research.google.com/github/petrosbeyene/pdf-data-extractor/blob/main/pdf_extractor_and_analyzer.ipynb) 
