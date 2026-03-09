# LangChain对于Agent的定义是“由大模型规划并自由组装各种链来满足用户需求”
import os
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# 以下依赖用于编写pdf创建写入相关代码
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime


# 将PlayWright智能体代码封装为工具函数
@tool
def summarize_website(url):
    """访问指定网站并返回内容总结"""
    try:
        # 初始化 Playwright 浏览器：
        sync_browser = create_sync_playwright_browser() # 创建同步执行的浏览器
        toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser) # 构建PlayWright浏览器工具
        tools = toolkit.get_tools() # 获取PlayWright浏览器操作函数

        # 初始化模型
        # 使用 智普 模型
        model = init_chat_model(
            # model="glm-5",
            model="glm-4.7",
            model_provider="openai", # 模型提供商，智普提供了openai请求格式的访问
            api_key='自己的apikey',
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
        )

        # 通过 LangChain Hub 拉取提示词模版
        prompt = hub.pull("hwchase17/openai-tools-agent")

        # 拉取提示词模板的代码与以下自定义提示词的代码等价:
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", "You are a helpful assistant"),
        #         ('placeholder': "{chat_history}"), #历史对话信息占位符，Agent会自动维护历史对话并填充到这个位置
        #         ("human", "{input}"),
        #         ("placeholder", "{agent_scratchpad}"),  # 工具内容占位信息，这部分agnet提示符不需要人工输入，同时也是写死的不可以修改
        #     ]
        # )

        # 通过 LangChain 创建 OpenAI 工具代理
        agent = create_openai_tools_agent(model, tools, prompt)

        # 通过 AgentExecutor 执行代理
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        # 定义任务
        command = {
            "input": "访问这个网站 https://liaoxuefeng.com/books/java/reflection/class/index.html 并帮我总结一下这个网站的内容"
        }

        # 执行任务
        response = agent_executor.invoke(command)
        return response.get("output", "未能获取网站内容总结")
    except Exception as e:
        return f"网站访问失败: {str(e)}"
    
#  创建PDF生成的工具函数
@tool
def generate_pdf(content):
    """将文本内容生成为PDF文件"""
    try:
        # 生成文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"website_summary_{timestamp}.pdf"

        # 创建PDF文档
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()

        # 注册中文字体（如果系统有的话）
        try:
            # Windows 系统字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            ]

            chinese_font_registered = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                        chinese_font_registered = True
                        print(f"✅ 成功注册中文字体: {font_path}")
                        break
                    except:
                        continue

            if not chinese_font_registered:
                print("⚠️ 未找到中文字体，使用默认字体")

        except Exception as e:
            print(f"⚠️ 字体注册失败: {e}")

        # 自定义样式 - 支持中文
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=30,
            fontName='ChineseFont' if 'chinese_font_registered' in locals() and chinese_font_registered else 'Helvetica-Bold'
        )

        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=12,
            fontName='ChineseFont' if 'chinese_font_registered' in locals() and chinese_font_registered else 'Helvetica'
        )

        # 构建PDF内容
        story = []

        # 标题
        story.append(Paragraph("网站内容总结报告", title_style))
        story.append(Spacer(1, 20))

        # 生成时间
        time_text = f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(time_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # 分隔线
        story.append(Paragraph("=" * 50, styles['Normal']))
        story.append(Spacer(1, 15))

        # 主要内容 - 改进中文处理
        if content:
            # 清理和处理内容
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            paragraphs = content.split('\n')

            for para in paragraphs:
                if para.strip():
                    # 处理特殊字符，确保PDF可以正确显示
                    clean_para = para.strip()
                    # 转换HTML实体
                    clean_para = clean_para.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

                    try:
                        story.append(Paragraph(clean_para, content_style))
                        story.append(Spacer(1, 8))
                    except Exception as para_error:
                        # 如果段落有问题，尝试用默认字体
                        try:
                            fallback_style = ParagraphStyle(
                                'Fallback',
                                parent=styles['Normal'],
                                fontSize=10,
                                leftIndent=20,
                                rightIndent=20,
                                spaceAfter=10
                            )
                            story.append(Paragraph(clean_para, fallback_style))
                            story.append(Spacer(1, 8))
                        except:
                            # 如果还是有问题，记录错误但继续
                            print(f"⚠️ 段落处理失败: {clean_para[:50]}...")
                            continue
        else:
            story.append(Paragraph("暂无内容", content_style))

        # 页脚信息
        story.append(Spacer(1, 30))
        story.append(Paragraph("=" * 50, styles['Normal']))
        story.append(Paragraph("本报告由 Playwright PDF Agent 自动生成", styles['Italic']))

        # 生成PDF
        doc.build(story)

        # 获取绝对路径
        abs_path = os.path.abspath(filename)
        print(f"  PDF文件生成完成: {abs_path}")
        return f"PDF文件已成功生成: {abs_path}"

    except Exception as e:
        error_msg = f"PDF生成失败: {str(e)}"
        print(error_msg)
        return error_msg
    
# 创建串行链
print("=== 创建串行链：网站总结 → PDF生成 ===")

simple_chain = summarize_website | generate_pdf


# 编写测试函数
def test_simple_chain(url):
    """测试简单串行链"""
    print(f"\n  开始处理URL: {url}")
    print("  步骤1: 网站总结...")
    print("  步骤2: 生成PDF...")

    result = simple_chain.invoke(url)
    print(f"✅ 完成: {result}")
    return result
    
if __name__ == "__main__":
     # 测试URL
    test_url = "https://www.microsoft.com/en-us/microsoft-365/blog/2025/01/16/copilot-is-now-included-in-microsoft-365-personal-and-family/?culture=zh-cn&country=cn"
    test_simple_chain(test_url)