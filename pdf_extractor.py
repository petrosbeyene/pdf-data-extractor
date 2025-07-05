"""
PDF Data Extractor for Investment Analysis
==========================================

This script extracts key investment information from earnings call transcripts
and financial reports for investor evaluation, focusing on:
- Growth prospects
- Business changes and strategic initiatives
- Key triggers and catalysts
- Material factors affecting future earnings and growth

Author: Petros Beyene Mola
"""

import re
import json
import os
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import datetime
from collections import defaultdict

# For Google Colab environment
# !pip install openai python-docx PyPDF2 nltk textstat

import openai
import PyPDF2
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import textstat

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

@dataclass
class InvestmentInsight:
    """Data class to store investment insights"""
    category: str
    insight: str
    confidence: float
    supporting_evidence: List[str]
    impact_level: str  # High, Medium, Low
    timeframe: str  # Short-term, Medium-term, Long-term
    source_location: Optional[str] = None  # Page or section reference

class PDFInvestmentAnalyzer:
    """
    Comprehensive analyzer for financial documents to extract key investment information
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the analyzer
        
        Args:
            openai_api_key: OpenAI API key for enhanced analysis (optional)
        """
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Investment-focused keywords and patterns
        self.growth_keywords = [
            'growth', 'expansion', 'scaling', 'increasing', 'accelerating',
            'growing', 'rise', 'boost', 'enhance', 'improve', 'strengthen',
            'momentum', 'trajectory', 'ramp-up', 'scale-up', 'outperform',
            'outpacing', 'market share', 'penetration', 'opportunity'
        ]
        
        self.financial_keywords = [
            'revenue', 'sales', 'turnover', 'ebitda', 'profit', 'margin',
            'pat', 'earnings', 'cash flow', 'roi', 'roce', 'roe', 'eps',
            'consolidated', 'guidance', 'forecast', 'outlook'
        ]
        
        self.risk_keywords = [
            'risk', 'challenge', 'concern', 'issue', 'problem', 'headwind',
            'pressure', 'decline', 'drop', 'fall', 'weak', 'slow', 'impact',
            'disruption', 'uncertainty', 'volatile', 'competition'
        ]
        
        self.acquisition_keywords = [
            'acquisition', 'merger', 'acquire', 'acquired', 'bought',
            'purchase', 'integration', 'synergy', 'consolidation', 'inorganic',
            'transformative', 'strategic', 'partnership', 'joint venture'
        ]
        
        self.guidance_keywords = [
            'guidance', 'target', 'expect', 'forecast', 'outlook',
            'projected', 'estimate', 'anticipate', 'confident', 'future',
            'next quarter', 'next year', 'long term', 'medium term'
        ]
        
        # Numerical pattern for extracting figures
        self.number_pattern = r'(\d+(?:\.\d+)?)\s*(?:crore|million|billion|%|percent)'
        
    # def extract_text_from_pdf(self, pdf_path: str) -> str:
    #     """
    #     Extract text from PDF file
        
    #     Args:
    #         pdf_path: Path to PDF file
            
    #     Returns:
    #         Extracted text content
    #     """
    #     try:
    #         with open(pdf_path, 'rb') as file:
    #             pdf_reader = PyPDF2.PdfReader(file)
    #             text = ""
    #             for page in pdf_reader.pages:
    #                 text += page.extract_text()
    #             return text
    #     except Exception as e:
    #         print(f"Error extracting PDF: {e}")
    #         return ""
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file using pdfplumber for better text quality

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting PDF with pdfplumber: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess the text with improved word separation
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text
        """
        # Remove contact info, headers, footers
        text = re.sub(r'Mumbai.*?\d{6}', '', text)
        text = re.sub(r'Symbol:.*?Code:', '', text)
        text = re.sub(r'Membership No\..*?Encl:', '', text)
        
        # Remove moderator instructions
        text = re.sub(r'Ladies and gentlemen.*?JM Financial\.', '', text)
        text = re.sub(r'As a reminder.*?phone\.', '', text)
        
        # Remove common transcript artifacts
        text = re.sub(r'Moderator:', '', text)
        text = re.sub(r'Page \d+', '', text)
        
        # Fix common word concatenation issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Insert space before capital letters
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Space between letters and numbers
        text = re.sub(r'([a-z])(of|to|in|on|at|by|for|with|and|the|a|an)', r'\1 \2', text)  # Common word boundaries
        
        # Fix specific financial terms
        text = re.sub(r'([a-z])(EBITDA|PAT|ROE|ROCE|YoY|QoQ)', r'\1 \2', text)
        text = re.sub(r'(EBITDA|PAT|ROE|ROCE|YoY|QoQ)([a-z])', r'\1 \2', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\%\$]', '', text)
        
        return text.strip()
    
    def is_meaningful_sentence(self, sentence: str) -> bool:
        """
        Check if a sentence contains meaningful business content
        
        Args:
            sentence: Sentence to evaluate
            
        Returns:
            Boolean indicating if sentence is meaningful
        """
        # Skip if mostly contact info, numbers, or formatting
        if any(pattern in sentence.lower() for pattern in [
            'membership no', 'symbol:', 'mumbai', 'department',
            'ladies and gentlemen', 'reminder', 'conference call',
            'analyst:', 'management:', 'moderator', 'thank you',
            'good day', 'welcome to', 'over to you'
        ]):
            return False
        
        # Skip very short sentences
        if len(sentence.split()) < 8:
            return False
        
        # Must contain actual business content
        business_indicators = ['revenue', 'growth', 'business', 'market', 'company', 
                              'acquisition', 'performance', 'margin', 'customer', 
                              'segment', 'quarter', 'year']
        return any(indicator in sentence.lower() for indicator in business_indicators)
    
    def extract_financial_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extract key financial metrics and numbers with context
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary of financial metrics with context
        """
        metrics = {}
        
        # Revenue patterns with context
        revenue_patterns = [
            r'(revenue.*?(\d+(?:\.\d+)?)\s*(?:crore|million|billion).*?)',
            r'(sales.*?(\d+(?:\.\d+)?)\s*(?:crore|million|billion).*?)',
            r'(turnover.*?(\d+(?:\.\d+)?)\s*(?:crore|million|billion).*?)'
        ]
        
        # Growth patterns with context
        growth_patterns = [
            r'(growth.*?(\d+(?:\.\d+)?)\s*(?:%|percent).*?)',
            r'(grew.*?(\d+(?:\.\d+)?)\s*(?:%|percent).*?)',
            r'(increase.*?(\d+(?:\.\d+)?)\s*(?:%|percent).*?)'
        ]
        
        # Margin patterns with context
        margin_patterns = [
            r'(margin.*?(\d+(?:\.\d+)?)\s*(?:%|percent).*?)',
            r'(ebitda.*?(\d+(?:\.\d+)?)\s*(?:%|percent).*?)'
        ]
        
        # Extract metrics with context
        for pattern in revenue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['revenue_context'] = [match[0][:100] for match in matches]  # First 100 chars of context
                metrics['revenue_figures'] = [match[1] for match in matches]
                
        for pattern in growth_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['growth_context'] = [match[0][:100] for match in matches]
                metrics['growth_rates'] = [match[1] for match in matches]
                
        for pattern in margin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['margin_context'] = [match[0][:100] for match in matches]
                metrics['margin_figures'] = [match[1] for match in matches]
        
        return metrics
    
    def identify_key_segments(self, text: str) -> Dict[str, str]:
        """
        Identify and extract key segments from the transcript
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary of key segments
        """
        segments = {}
        
        # Split by common section headers
        sections = [
            ('opening_remarks', r'opening remarks?|opening comment'),
            ('financial_highlights', r'financial performance|financial highlights'),
            ('business_update', r'business.*?update|business.*?performance'),
            ('outlook', r'outlook|future|guidance|going forward'),
            ('qa_section', r'question.*?answer|q&a|questions')
        ]
        
        for section_name, pattern in sections:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.start()
                # Find next section or end of text
                next_sections = [re.search(p, text[start+100:], re.IGNORECASE) for _, p in sections]
                next_sections = [m for m in next_sections if m is not None]
                
                if next_sections:
                    end = min(m.start() for m in next_sections) + start + 100
                    segments[section_name] = text[start:end]
                else:
                    segments[section_name] = text[start:]
        
        return segments
    
    def extract_growth_drivers(self, text: str) -> List[InvestmentInsight]:
        """
        Extract growth drivers and prospects
        
        Args:
            text: Text content
            
        Returns:
            List of growth-related insights
        """
        insights = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Skip non-meaningful sentences
            if not self.is_meaningful_sentence(sentence):
                continue
            
            # Check for growth-related content
            if any(keyword in sentence.lower() for keyword in self.growth_keywords):
                # Extract numerical growth figures
                growth_numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', sentence)
                
                # Categorize by impact level
                impact_level = "Medium"
                if growth_numbers:
                    if any(float(num) > 20 for num in growth_numbers):
                        impact_level = "High"
                    elif any(float(num) < 10 for num in growth_numbers):
                        impact_level = "Low"
                
                # Determine timeframe
                timeframe = "Medium-term"
                if any(term in sentence.lower() for term in ['this year', 'fy24', 'quarter']):
                    timeframe = "Short-term"
                elif any(term in sentence.lower() for term in ['long term', 'future', 'next 3']):
                    timeframe = "Long-term"
                
                insight = InvestmentInsight(
                    category="Growth Drivers",
                    insight=sentence.strip(),
                    confidence=0.8,
                    supporting_evidence=growth_numbers,
                    impact_level=impact_level,
                    timeframe=timeframe
                )
                insights.append(insight)
        
        return insights[:10]  # Return top 10 insights
    
    def extract_business_changes(self, text: str) -> List[InvestmentInsight]:
        """
        Extract key business changes and strategic initiatives
        
        Args:
            text: Text content
            
        Returns:
            List of business change insights
        """
        insights = []
        sentences = sent_tokenize(text)
        
        change_indicators = [
            'new', 'launch', 'introduce', 'expand', 'acquire', 'partnership',
            'strategic', 'initiative', 'transformation', 'restructure'
        ]
        
        for sentence in sentences:
            # Skip non-meaningful sentences
            if not self.is_meaningful_sentence(sentence):
                continue
            
            if any(indicator in sentence.lower() for indicator in change_indicators):
                # Assess impact based on keywords
                impact_level = "Medium"
                if any(keyword in sentence.lower() for keyword in self.acquisition_keywords):
                    impact_level = "High"
                elif 'new customer' in sentence.lower() or 'new product' in sentence.lower():
                    impact_level = "High"
                
                # Determine timeframe based on context
                timeframe = "Medium-term"
                if "completed" in sentence.lower() or "recently" in sentence.lower():
                    timeframe = "Short-term"
                elif "plan" in sentence.lower() or "future" in sentence.lower():
                    timeframe = "Long-term"
                
                insight = InvestmentInsight(
                    category="Business Changes",
                    insight=sentence.strip(),
                    confidence=0.7,
                    supporting_evidence=[],
                    impact_level=impact_level,
                    timeframe=timeframe
                )
                insights.append(insight)
        
        return insights[:10]
    
    def extract_financial_guidance(self, text: str) -> List[InvestmentInsight]:
        """
        Extract financial guidance and forward-looking statements
        
        Args:
            text: Text content
            
        Returns:
            List of guidance insights
        """
        insights = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Skip non-meaningful sentences
            if not self.is_meaningful_sentence(sentence):
                continue
            
            if any(keyword in sentence.lower() for keyword in self.guidance_keywords):
                # Extract numerical guidance
                numbers = re.findall(self.number_pattern, sentence)
                
                # Determine timeframe based on context
                timeframe = "Short-term"
                if any(term in sentence.lower() for term in ['long term', 'future', 'years']):
                    timeframe = "Long-term"
                elif any(term in sentence.lower() for term in ['medium term', 'next year']):
                    timeframe = "Medium-term"
                
                insight = InvestmentInsight(
                    category="Financial Guidance",
                    insight=sentence.strip(),
                    confidence=0.9,
                    supporting_evidence=numbers,
                    impact_level="High",
                    timeframe=timeframe
                )
                insights.append(insight)
        
        return insights[:5]
    
    def extract_risks_and_challenges(self, text: str) -> List[InvestmentInsight]:
        """
        Extract risks and challenges mentioned
        
        Args:
            text: Text content
            
        Returns:
            List of risk insights
        """
        insights = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Skip non-meaningful sentences
            if not self.is_meaningful_sentence(sentence):
                continue
            
            if any(keyword in sentence.lower() for keyword in self.risk_keywords):
                # Determine impact level based on language intensity
                impact_level = "Medium"
                high_intensity = ['significant', 'major', 'substantial', 'critical', 'severe']
                if any(word in sentence.lower() for word in high_intensity):
                    impact_level = "High"
                
                insight = InvestmentInsight(
                    category="Risks & Challenges",
                    insight=sentence.strip(),
                    confidence=0.6,
                    supporting_evidence=[],
                    impact_level=impact_level,
                    timeframe="Short-term"
                )
                insights.append(insight)
        
        return insights[:5]
    
    def analyze_acquisition_impact(self, text: str, acquisition_name: str = None) -> Dict[str, List[str]]:
        """
        Analyze the impact of a specific acquisition
        
        Args:
            text: Text content
            acquisition_name: Name of the acquisition to focus on (optional)
            
        Returns:
            Dictionary with acquisition analysis
        """
        acquisition_info = {
            'acquisition_details': [],
            'financial_impact': [],
            'strategic_benefits': [],
            'integration_plan': []
        }
        
        if not acquisition_name:
            # Improved acquisition name detection
            acquisition_patterns = [
                r'(acquisition of|acquired|bought)\s+([A-Z][A-Za-z\s]+(?:India|Limited|Corp|Inc|Pack)?)',
                r'(transformative acquisition of)\s+([A-Z][A-Za-z\s]+)',
                r'(acquiring)\s+([A-Z][A-Za-z\s]+(?:India|Limited|Corp|Inc|Pack)?)'
            ]
            
            for pattern in acquisition_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    acquisition_name = matches[0][1].strip()
                    break
        
        if not acquisition_name:
            return acquisition_info
            
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if acquisition_name.lower() in sentence.lower() and self.is_meaningful_sentence(sentence):
                sentence_lower = sentence.lower()
                
                if any(keyword in sentence_lower for keyword in ['acquired', 'acquisition', 'bought']):
                    acquisition_info['acquisition_details'].append(sentence.strip())
                
                elif any(keyword in sentence_lower for keyword in ['revenue', 'sales', 'margin', 'growth']):
                    acquisition_info['financial_impact'].append(sentence.strip())
                
                elif any(keyword in sentence_lower for keyword in ['synergy', 'customer', 'technology', 'capability']):
                    acquisition_info['strategic_benefits'].append(sentence.strip())
                
                elif any(keyword in sentence_lower for keyword in ['integration', 'plan', 'consolidate']):
                    acquisition_info['integration_plan'].append(sentence.strip())
        
        return acquisition_info
    
    def generate_summary_with_gpt(self, text: str) -> str:
        """
        Generate executive summary using GPT (if API key provided)
        
        Args:
            text: Full text content
            
        Returns:
            Executive summary
        """
        if not self.openai_api_key:
            return "GPT summary not available - no API key provided"
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            prompt = f"""
            Analyze this earnings call transcript and provide a concise executive summary 
            focusing on key investment highlights:
            
            1. Major financial performance metrics
            2. Key business developments and changes
            3. Growth drivers and opportunities
            4. Forward guidance and outlook
            5. Key risks or challenges mentioned
            
            Transcript: {text[:4000]}...
            
            Provide a structured summary in 300-400 words that would be valuable for an investor.
            Focus on material information that could affect future earnings and growth.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at summarizing earnings calls for investors."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating GPT summary: {e}"
    
    def analyze_document(self, text_content: str, company_name: str = None) -> Dict[str, Any]:
        """
        Main analysis function that extracts all key investment information
        
        Args:
            text_content: Full text content of earnings call or financial report
            company_name: Name of the company (optional)
            
        Returns:
            Comprehensive analysis results
        """
        # Preprocess text
        clean_text = self.preprocess_text(text_content)
        
        # Extract different types of insights
        growth_insights = self.extract_growth_drivers(clean_text)
        business_changes = self.extract_business_changes(clean_text)
        financial_guidance = self.extract_financial_guidance(clean_text)
        risks = self.extract_risks_and_challenges(clean_text)
        
        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics(clean_text)
        
        # Identify key segments
        segments = self.identify_key_segments(clean_text)
        
        # Analyze acquisitions
        acquisition_analysis = self.analyze_acquisition_impact(clean_text)
        
        # Generate summary
        executive_summary = self.generate_summary_with_gpt(clean_text)
        
        # Compile results
        results = {
            'company_name': company_name,
            'executive_summary': executive_summary,
            'financial_metrics': financial_metrics,
            'growth_insights': [insight.__dict__ for insight in growth_insights],
            'business_changes': [insight.__dict__ for insight in business_changes],
            'financial_guidance': [insight.__dict__ for insight in financial_guidance],
            'risks_challenges': [insight.__dict__ for insight in risks],
            'acquisition_analysis': acquisition_analysis,
            'key_segments': segments,
            'analysis_metadata': {
                'total_insights': len(growth_insights + business_changes + financial_guidance + risks),
                'text_length': len(clean_text),
                'readability_score': textstat.flesch_reading_ease(clean_text),
                'analysis_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        return results
    
    def generate_investment_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a formatted investment analysis report
        
        Args:
            analysis_results: Results from analyze_document
            
        Returns:
            Formatted report string
        """
        report = []
        company_name = analysis_results.get('company_name', 'Company')
        report.append("=" * 80)
        report.append(f"{company_name.upper()} - INVESTMENT ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Executive Summary
        report.append("\n📊 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        report.append(analysis_results['executive_summary'])
        
        # Financial Metrics
        report.append("\n💰 KEY FINANCIAL METRICS")
        report.append("-" * 50)
        for metric, values in analysis_results['financial_metrics'].items():
            if values:
                report.append(f"{metric.replace('_', ' ').title()}: {', '.join(values)}")
        
        # Acquisition Analysis
        if any(analysis_results['acquisition_analysis'].values()):
            report.append("\n🔄 ACQUISITION ANALYSIS")
            report.append("-" * 50)
            for category, items in analysis_results['acquisition_analysis'].items():
                if items:
                    report.append(f"\n{category.replace('_', ' ').title()}:")
                    for item in items[:3]:
                        report.append(f"• {item}")
        
        # Growth Insights
        report.append("\n🚀 GROWTH DRIVERS & PROSPECTS")
        report.append("-" * 50)
        for insight in analysis_results['growth_insights']:
            report.append(f"• {insight['insight']}")
            if insight['supporting_evidence']:
                report.append(f"  Evidence: {', '.join(insight['supporting_evidence'])}")
            report.append(f"  Impact: {insight['impact_level']} | Timeframe: {insight['timeframe']}")
            report.append("")
        
        # Business Changes
        report.append("\n🔄 KEY BUSINESS CHANGES")
        report.append("-" * 50)
        for insight in analysis_results['business_changes']:
            report.append(f"• {insight['insight']}")
            report.append(f"  Impact: {insight['impact_level']} | Timeframe: {insight['timeframe']}")
            report.append("")
        
        # Financial Guidance
        report.append("\n🎯 FINANCIAL GUIDANCE & OUTLOOK")
        report.append("-" * 50)
        for insight in analysis_results['financial_guidance']:
            report.append(f"• {insight['insight']}")
            if insight['supporting_evidence']:
                report.append(f"  Numbers: {', '.join(insight['supporting_evidence'])}")
            report.append(f"  Timeframe: {insight['timeframe']}")
            report.append("")
        
        # Risks & Challenges
        report.append("\n⚠️ RISKS & CHALLENGES")
        report.append("-" * 50)
        for insight in analysis_results['risks_challenges']:
            report.append(f"• {insight['insight']}")
            report.append(f"  Impact: {insight['impact_level']}")
            report.append("")
        
        # Investment Thesis
        report.append("\n💡 INVESTMENT THESIS")
        report.append("-" * 50)
        # Generate investment thesis points based on the analysis
        thesis_points = []
        
        # Add growth-related thesis points
        if analysis_results['growth_insights']:
            high_impact_growth = [i for i in analysis_results['growth_insights'] if i['impact_level'] == 'High']
            if high_impact_growth:
                thesis_points.append(f"Strong growth potential identified in key areas")
        
        # Add acquisition-related thesis points
        if analysis_results['acquisition_analysis']['strategic_benefits']:
            thesis_points.append(f"Strategic acquisitions enhancing capabilities and market position")
        
        # Add guidance-related thesis points
        if analysis_results['financial_guidance']:
            thesis_points.append(f"Management provides positive forward guidance")
        
        # Add risk-related thesis points
        if analysis_results['risks_challenges']:
            thesis_points.append(f"Key risks to monitor include: {analysis_results['risks_challenges'][0]['insight'][:50]}...")
        
        for point in thesis_points:
            report.append(f"✓ {point}")
        
        # Metadata
        report.append("\n📈 ANALYSIS METADATA")
        report.append("-" * 50)
        metadata = analysis_results['analysis_metadata']
        report.append(f"Total Insights Extracted: {metadata['total_insights']}")
        report.append(f"Document Length: {metadata['text_length']} characters")
        report.append(f"Readability Score: {metadata['readability_score']:.1f}")
        report.append(f"Analysis Date: {metadata['analysis_date']}")
        
        return "\n".join(report)

def main():
    """
    Main function to demonstrate the usage
    """
    # Initialize analyzer
    # get API key from environment or user input
    # import os
    # api_key = os.getenv('OPENAI_API_KEY') or input('Enter your OpenAI API key: ')
    # analyzer = PDFInvestmentAnalyzer(openai_api_key=api_key)
    
    # For demo without GPT
    analyzer = PDFInvestmentAnalyzer()
    
    # For Google Colab, download the file first
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Adjust path as needed based on your Google Drive structure
    # pdf_path = "/content/drive/MyDrive/your_file_path.pdf"
    pdf_path = "/content/drive/MyDrive/PDFs/SJS_Transcript_Call.pdf"
    
    # Extract text from PDF
    text_content = analyzer.extract_text_from_pdf(pdf_path)
    
    # If PDF extraction fails, use sample text for demonstration
    if not text_content:
        print("Using sample text for demonstration...")
        text_content = """
        SJS Enterprises Limited Q1 FY2024 Earnings Conference Call July 27, 2023
        
        K.A. Joseph: We have completed the transformative acquisition of Walter Pack India within the set timeline.
        After the successful acquisition of Exotech and the robust performance we delivered there in the last two years, 
        we have gained more confidence in our execution capabilities of acquiring and integrating companies that could 
        take SJS to the next level of growth.
        
        Walter Pack acquisition has opened up a plethora of new opportunities for us. With this acquisition, we have 
        penetrated deeper in passenger vehicles and consumer appliances segment, thereby further reducing our two 
        wheeler dependence. According to SJS Q1 FY24 pro forma numbers, which includes Walter Pack India, 36% revenue 
        contribution would be from two wheelers, 36% from passenger vehicles and 28% from consumer appliances and others.
        
        We have seen a good start for Walter Pack India this quarter. Walter Pack Q1 witnessed a strong revenue growth 
        of 21% YoY and a robust margin performance with EBITDA margins around 31.5%.
        """
    
    # Analyze the document
    results = analyzer.analyze_document(text_content, company_name="SJS Enterprises")
    
    # Generate report
    report = analyzer.generate_investment_report(results)
    
    # Print report
    print(report)
    
    # Save results to JSON for further analysis
    with open('investment_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nAnalysis complete! Results saved to 'investment_analysis.json'")
    
    # Create a pandas DataFrame for key metrics
    metrics_df = pd.DataFrame({
        'Category': ['Growth', 'Financial', 'Risk', 'Strategic'],
        'Insights Count': [
            len(results['growth_insights']),
            len(results['financial_guidance']),
            len(results['risks_challenges']),
            len(results['business_changes'])
        ]
    })
    
    # Display the DataFrame
    print("\nInsights Distribution:")
    print(metrics_df)
    
    # Return the results for further analysis in the notebook
    return results

if __name__ == "__main__":
    main() 