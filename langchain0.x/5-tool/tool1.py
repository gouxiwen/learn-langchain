import pandas as pd 
from langchain_experimental.tools import PythonAstREPLTool # 从LangChain依赖库引入Python代码解释器

df = pd.read_csv('global_cities_data.csv')
tool = PythonAstREPLTool(locals={"df": df}) # 传递给代码解释器的局部变量，这里是读取表格内容的pandas对象
res = tool.invoke("df['GDP_Billion_USD'].mean()") # 计算变量GDP的均值
# res = tool.invoke({'query': 'import pandas as pd\nmax_gdp = df[\'GDP_Billion_USD\'].max()\ncountry_with_max_gdp = df[df[\'GDP_Billion_USD\'] == max_gdp][\'Country\'].values[0]\nprint(f"最大GDP的国家是 {country_with_max_gdp}，其GDP为 {max_gdp} 亿美元。")'}) # 计算变量GDP的均值

print(res)