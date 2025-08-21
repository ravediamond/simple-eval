import io
from datetime import datetime
from typing import List, Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF

class EvaluationPDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        # Custom styles for the PDF
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubTitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER,
            spaceAfter=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#ff6b35'),
            spaceBefore=20,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='InsightText',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            leftIndent=20,
            spaceBefore=6,
            spaceAfter=6
        ))
        
        self.styles.add(ParagraphStyle(
            name='AppPromo',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER,
            borderColor=colors.HexColor('#ff6b35'),
            borderWidth=1,
            borderPadding=10,
            backColor=colors.HexColor('#fff5f2')
        ))
        
    def create_score_chart(self, distribution: Dict[str, int]):
        """Create a bar chart for score distribution"""
        drawing = Drawing(400, 200)
        
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.height = 120
        chart.width = 300
        
        chart.data = [[
            distribution.get('excellent', 0),
            distribution.get('good', 0),
            distribution.get('needs_improvement', 0),
            distribution.get('poor', 0)
        ]]
        
        chart.categoryAxis.categoryNames = ['Excellent\n(90%+)', 'Good\n(70-89%)', 'Needs Work\n(50-69%)', 'Poor\n(<50%)']
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(distribution.values()) + 1 if distribution.values() else 5
        
        chart.bars[0].fillColor = colors.HexColor('#28a745')
        chart.bars[1].fillColor = colors.HexColor('#17a2b8')
        chart.bars[2].fillColor = colors.HexColor('#ffc107')
        chart.bars[3].fillColor = colors.HexColor('#dc3545')
        
        drawing.add(chart)
        return drawing
    
    def generate_pdf(self, results: List[Any], insights: Dict[str, Any], filename: str, evaluation_time: str) -> io.BytesIO:
        """Generate a comprehensive PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        story = []
        
        # Header with App Branding
        story.append(Paragraph("🚀 EvalNow", self.styles['CustomTitle']))
        story.append(Paragraph("AI Response Evaluation Report", self.styles['SubTitle']))
        story.append(Spacer(1, 20))
        
        # Report Info
        report_info = [
            ["File:", filename],
            ["Generated:", evaluation_time + " UTC"],
            ["Total Questions:", str(insights['total_questions'])],
            ["Report Type:", "Comprehensive AI Evaluation Analysis"]
        ]
        
        info_table = Table(report_info, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#333333')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("📊 Executive Summary", self.styles['SectionHeader']))
        
        # Key Metrics Table
        avg_score = f"{insights['average_score'] * 100:.1f}%"
        pass_rate = f"{insights['pass_rate'] * 100:.1f}%"
        
        metrics_data = [
            ["Metric", "Value", "Assessment"],
            ["Average Score", avg_score, "Good" if insights['average_score'] >= 0.7 else "Needs Improvement"],
            ["Pass Rate (≥70%)", pass_rate, "Excellent" if insights['pass_rate'] >= 0.8 else "Good" if insights['pass_rate'] >= 0.6 else "Poor"],
            ["Total Questions", str(insights['total_questions']), "Comprehensive Sample" if insights['total_questions'] >= 10 else "Limited Sample"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff6b35')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Score Distribution Chart
        if 'distribution' in insights:
            story.append(Paragraph("📈 Score Distribution", self.styles['SectionHeader']))
            chart = self.create_score_chart(insights['distribution'])
            story.append(chart)
            story.append(Spacer(1, 20))
        
        # AI Analysis
        if 'ai_analysis' in insights:
            ai_analysis = insights['ai_analysis']
            
            story.append(Paragraph("🤖 AI-Powered Analysis", self.styles['SectionHeader']))
            story.append(Paragraph(ai_analysis.get('summary', ''), self.styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Key Insights
            story.append(Paragraph("💡 Key Insights", self.styles['Heading3']))
            for insight in ai_analysis.get('key_insights', []):
                story.append(Paragraph(f"• {insight}", self.styles['InsightText']))
            story.append(Spacer(1, 15))
            
            # Recommendations
            story.append(Paragraph("🎯 Recommendations", self.styles['Heading3']))
            for recommendation in ai_analysis.get('recommendations', []):
                story.append(Paragraph(f"• {recommendation}", self.styles['InsightText']))
            story.append(Spacer(1, 20))
        
        # Detailed Results Table (Top 10 or all if <= 10)
        story.append(Paragraph("📋 Detailed Results", self.styles['SectionHeader']))
        
        # Limit to top 10 for PDF readability
        display_results = results[:10] if len(results) > 10 else results
        
        results_data = [["#", "Question", "Score", "Assessment"]]
        
        for i, result in enumerate(display_results, 1):
            score_percent = int(result.score * 100)
            if score_percent >= 90:
                assessment = "Excellent"
            elif score_percent >= 70:
                assessment = "Good"
            elif score_percent >= 50:
                assessment = "Needs Work"
            else:
                assessment = "Poor"
            
            # Truncate long questions for PDF
            question = result.question[:80] + "..." if len(result.question) > 80 else result.question
            
            results_data.append([
                str(i),
                question,
                f"{score_percent}%",
                assessment
            ])
        
        if len(results) > 10:
            results_data.append(["...", f"+ {len(results) - 10} more questions", "", "See full results online"])
        
        results_table = Table(results_data, colWidths=[0.5*inch, 3.5*inch, 0.8*inch, 1.2*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#333333')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # Question numbers centered
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),  # Scores centered
            ('ALIGN', (3, 0), (3, -1), 'CENTER'),  # Assessment centered
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fafafa')])
        ]))
        story.append(results_table)
        story.append(Spacer(1, 30))
        
        # App Promotion Footer
        story.append(Paragraph(
            """
            <b>🚀 Generated by EvalNow</b><br/>
            EvalNow is the fastest way to evaluate your AI responses. Upload your data, get AI-powered insights instantly!<br/>
            <b>Try it yourself:</b> https://evalnow.xyz | No signup required • Privacy-first • AI-powered analysis<br/>
            <i>Perfect for developers, researchers, and AI teams who need quick, reliable evaluation results.</i>
            """,
            self.styles['AppPromo']
        ))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer