# langchain和langgraph学习项目

目前langchain最新版本是v1.x，而参考的教程使用的是v0.x，虽然api过时，但可以通过学习了解langchain的发展及基础知识，对未来使用langchain v1.x版本也有帮助。

教程地址： https://www.zhihu.com/column/c_1928432715688019821

## 虚拟环境

使用anaconda管理虚拟环境

查看conda版本

conda --version

创建虚拟环境

conda create -n langchainenv python=3.12 #创建langchain开发环境

激活虚拟环境
使用如下命令即可激活创建的虚拟环境。

conda activate langchainenv

此时使用python --version可以检查当前python版本是否为所想要的（即虚拟环境的python版本）。

在4.6版本以前需要使用如下命令：

Linux: source activate langchainenv
Windows（cmd）: activate langchainenv

配置镜像

单次使用
pip install gradio==5.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

全局配置

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

## langchain版本

由于参考的教程使用v0.x，而最新版本是v1.x，所以使用langchain_classic，而不是langchain最新版本

pip install langchain-classic 或者 pip install langchain==0.3.26

pip show langchain查看我们安装的langchain版本

## langgraph

创建虚拟环境

conda create -n langgraphenv python=3.12 #创建langgraph开发环境

激活虚拟环境
使用如下命令即可激活创建的虚拟环境。

conda activate langgraphenv

此时使用python --version可以检查当前python版本是否为所想要的（即虚拟环境的python版本）。

在4.6版本以前需要使用如下命令：

Linux: source activate langgraphenv
Windows（cmd）: activate langgraphenv

安装langgraph相关依赖，都需要是0.x版本

查看可用版本pip index versions 包名

pip install langgraph==0.6.6

pip install langchain==0.3.26

langchain-core==0.3.83

pip install langchain-openai==0.3.35

### depoly

使用LangSmith、LangGraph Studio 和 LangGraph CLI搭建和部署、调试langgraph应用

LangSmith是需要付费的，但对于个人开发者有免费额度，可以用于测试和学习。

安装依赖

pip install -r requirements.txt

启动应用

langgraph dev

使用agent-chat-ui在前端展示

```
git clone https://github.com/langchain-ai/agent-chat-ui.git //将agent-chat-ui 拉取到本地
cd agent-chat-ui // 进入agent-chat-ui项目目录
pnpm install // 安装agent-chat-ui相关依赖
pnpm run dev // 运行agent-chat-ui项目
```

打开页面后配置响应的langgraph部署地址及graph ID（graph.json中graphs中的配置），即可使用
