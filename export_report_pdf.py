"""Export the evidence report to a self-contained PDF with wrapped tables/figures."""
import argparse
from html import escape
from pathlib import Path
import markdown2
from bs4 import BeautifulSoup,Tag
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether,Preformatted
import matplotlib

ROOT=Path(__file__).resolve().parent

def export(source,destination):
    fonts=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    for name,file in [('Lab','DejaVuSans.ttf'),('Lab-Bold','DejaVuSans-Bold.ttf'),('Lab-Italic','DejaVuSans-Oblique.ttf'),('LabMono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Lab',normal='Lab',bold='Lab-Bold',italic='Lab-Italic',boldItalic='Lab-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='LabBody',fontName='Lab',fontSize=8.5,leading=12,spaceAfter=7,splitLongWords=True))
    styles.add(ParagraphStyle(name='LabCell',fontName='Lab',fontSize=6.5,leading=9,splitLongWords=True))
    for name,size in [('Heading1',17),('Heading2',12),('Heading3',10)]:
        styles[name].fontName='Lab-Bold';styles[name].fontSize=size;styles[name].leading=size+4;styles[name].spaceBefore=10;styles[name].keepWithNext=True
    styles['Code'].fontName='LabMono';styles['Code'].fontSize=6.5;styles['Code'].leading=9
    width=A4[0]-72
    def inline(node):
        if not isinstance(node,Tag):return escape(str(node))
        text=''.join(inline(child) for child in node.children)
        if node.name in ['strong','b']:return '<b>'+text+'</b>'
        if node.name in ['em','i']:return '<i>'+text+'</i>'
        if node.name=='code':return '<font name="LabMono">'+text+'</font>'
        if node.name=='br':return '<br/>'
        if node.name=='a' and str(node.get('href','')).startswith('http'):
            return '<link href="'+escape(node['href'],quote=True)+'" color="#205d8e">'+text+'</link>'
        return text
    soup=BeautifulSoup(markdown2.markdown(source.read_text(encoding='utf-8'),extras=['tables','fenced-code-blocks']),'html.parser')
    story=[]
    for node in soup.children:
        if not isinstance(node,Tag):continue
        if node.name in ['h1','h2','h3']:
            story.append(Paragraph(inline(node),styles['Heading'+node.name[-1]]))
        elif node.name=='table':
            data=[]
            for row in node.find_all('tr'):
                data.append([Paragraph(('<b>'+inline(cell)+'</b>') if cell.name=='th' else inline(cell),styles['LabCell']) for cell in row.find_all(['th','td'],recursive=False)])
            count=len(data[0]);ratios=[1]*count
            header=[c.get_text() for c in node.find('tr').find_all(['th','td'])]
            for index,value in enumerate(header):
                if any(word in value.lower() for word in ['technique','configuration','objective','diagnosis','exact tweet','text']):ratios[index]=2.2
            if count==3 and 'Exact tweet' in header:ratios=[.65,2.6,2.7]
            widths=[width*r/sum(ratios) for r in ratios]
            tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),
              ('VALIGN',(0,0),(-1,-1),'TOP'),('BOX',(0,0),(-1,-1),.35,colors.HexColor('#a8b3bf')),
              ('INNERGRID',(0,0),(-1,-1),.2,colors.HexColor('#c5cdd5')),
              ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
              ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.extend([tab,Spacer(1,9)])
        elif node.name in ['ul','ol']:
            for i,item in enumerate(node.find_all('li',recursive=False)):
                story.append(Paragraph(('• ' if node.name=='ul' else f'{i+1}. ')+inline(item),styles['LabBody']))
        elif node.find('img'):
            for img in node.find_all('img'):
                path=source.parent/img['src']
                if not path.exists():raise FileNotFoundError(path)
                iw,ih=PILImage.open(path).size;scale=min(width/iw,310/ih)
                story.extend([Image(str(path),width=iw*scale,height=ih*scale),Spacer(1,9)])
        elif node.name=='pre':story.append(Preformatted(node.get_text(),styles['Code'],maxLineLength=100))
        else:story.append(Paragraph(inline(node),styles['LabBody']))
    def page(canvas,document):
        canvas.saveState();canvas.setFont('Lab',7);canvas.setFillColor(colors.HexColor('#53616d'))
        canvas.drawString(36,22,'HW4.5 · RTX 4090 · Single-GPU submission')
        canvas.drawRightString(A4[0]-36,22,str(document.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36,
       title='HW4.5 — RTX 4090 Training Throughput and Memory Engineering',author='GPU lab submission')
    doc.build(story,onFirstPage=page,onLaterPages=page)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'REPORT.md');p.add_argument('--output',type=Path,default=ROOT/'REPORT.pdf');args=p.parse_args();export(args.source,args.output)
