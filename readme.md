# langchain和langgraph学习项目

目前langchain最新版本是v1.x，而参考的教程使用的是从v0.x开始过渡到v1.x，虽然api过时，但可以通过学习了解langchain的发展及基础知识，对未来使用langchain v1.x版本也有帮助。

教程地址： https://www.zhihu.com/column/c_1928432715688019821
langchain文档： https://docs.langchain.com/oss/python/langgraph/overview
langchain中文文档： https://docs.langchain.org.cn/langsmith/home

## 虚拟环境

使用anaconda管理虚拟环境

安装anaconda

https://docs.conda.io/en/latest/miniconda.html

注意：自 2025 年 8 月 15 日起，Anaconda 已停止为 Intel Mac 电脑（OS X 64 位）构建软件包。现有的 Intel (MacOS X 86_64) 安装程序仍可在 https://repo.anaconda.com/miniconda/ 获取。最后一个适用于 Intel Mac 电脑的 Miniconda 安装程序版本为 25.7.x。

查看conda版本

conda --version

创建虚拟环境

conda create -n xxx python=3.12

激活虚拟环境
使用如下命令即可激活创建的虚拟环境。

conda activate xxx

此时使用python --version可以检查当前python版本是否为所想要的（即虚拟环境的python版本）。

conda在4.6版本以前需要使用如下命令：

Linux: conda activate xxx
Windows（cmd）: activate xxx

配置镜像

单次使用
pip install gradio==5.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

全局配置

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

## langchain版本

### langchain0.x

由于参考的教程使用v0.x，而最新版本是v1.x，所以使用langchain_classic，而不是langchain最新版本

创建虚拟环境

conda create -n langchain-classic-env python=3.12

安装langchain0.x相关依赖，都需要是0.x版本

pip install langchain-classic 或者 pip install langchain==0.3.26

pip show langchain查看我们安装的langchain版本

### langchain1.x

**倒反天罡**

LangChain 1.0 已不再作为 LangGraph 的基座，反而其智能体构建于 LangGraph 之上，LangGraph更底层。

在 1.0 时代，官方建议如下框架选型策略：

如需快速构建智能体与自主应用程序，推荐使用 LangChain；

若需结合确定性流程与智能工作流、实现深度定制、精确控制响应延迟，则应使用底层编排框架 LangGraph。

创建虚拟环境

conda create -n langchain-env python=3.12

## langgraph

### langgraph0.x

创建虚拟环境

conda create -n langgraph-classic-env python=3.12

安装langgraph0.x相关依赖，都需要是0.x版本

查看可用版本pip index versions 包名

pip install langgraph==0.6.6 langchain==0.3.26 langchain-core==0.3.83 langchain-openai==0.3.35

### langgraph1.x

创建虚拟环境

conda create -n langgraph-env python=3.12

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

### 长短期记忆管理

#### 短记忆

InMemorySaver

短期记忆与线程相关，在与智能体对话时需要携带config线程id信息，根据线程id区分会话，默认存储在内存中

短记忆持久化

需要安装postgres数据库

使用langchain-checkpoint-postgres管理短期记忆数据库持久化存储到PostgreSQL数据库中

pip install "psycopg[binary,pool]" langgraph-checkpoint-postgres==2.0.23

#### 长期记忆

InMemoryStore

LangGraph 长期记忆常用于构建持久的知识体系，实现跨线程的学习能力

长记忆持久化

需要安装postgres数据库

使用LangGraph提供了PostgresStore工具类，用于将长期记忆持久化存储到PostgreSQL数据库中

#### 持久化选择其他数据库

对于短期记忆checkpoint来说，无论是InMemorySaver还是PostgresSaver都是通过继承BaseCheckpointSaver抽象类并定义存档点接口实现的。除此之外，LangGraph还基于BaseCheckpointSaver抽象类定义了SqliteSaver。如果有接入内部自定义存储系统的需求，可以通过继承BaseCheckpointSaver抽象类并自定义相关方法，相关文档见： https://reference.langchain.org.cn/python/langgraph/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver

同样的LangGraph长期记忆使用的InMemoryStore、PostgresStore是通过继承BaseStore抽象类来实现的，上面提到的put, get和set方法都是BaseStore抽象类中的接口方法，用户可以自定义相关方法实现长期记忆的定制，相关文档见: https://reference.langchain.org.cn/python/langgraph/store/#langgraph.store.base

## deepagents

创建虚拟环境

conda create -n deepagents-env python=3.12

## rag

基于langchain的多模态RAG系统
