[TOC]

# 项目概述

这是一个基于 LangGraph 的金融分析 Agent 系统，用于分析 A 股股票。系统包含五个 Agent：基本面分析 Agent、技术分析Agent、估值分析 Agent、新闻分析 Agent 和总结 Agent。前四个 Agent 通过 MCP 工具获取 A 股相关数据并与大语言模型（LLM）交互；总结 Agent 综合上游数据，提供最终投资建议。

**This content is only supported in a Feishu Docs**

# 项目运行

1. 安装依赖

```Plain
export PYTHONPATH=./                   #把当前目录加入 Python 的包搜索路径
pip install -r requirements.txt
```

2. 设置API

项目支持两种模式：**API 调用大模型**和​**本地FinR1模型**​。

（FinR1解读见[金融推理大模型Fin-r1 ](https://y95yb64h04.feishu.cn/wiki/AiaLwa3Q4iEI7rkXmY6cCOKUnvf?from=from_copylink)，用到了sft和grpo，后面用来面试的时候会提及）

注意：目前只有Summary Agent可以使用本地 FinR1，其余4个Agent仍然依赖API调用大模型，所以无论哪种模式，都必须配置 API\_KEY。

USE\_LOCAL\_MODEL=api表示使用api调用大模型，如果想使用FinR1，请设置为USE\_LOCAL\_MODEL=local，并自行下载https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1模型（大小为7B）。

```Plain
cd Financial-MCP-Agent/                         #进入本项目的目录
cp .env.example .env                            #把示例配置文件复制一份，命名为 .env，之后你要修改 .env 文件里的参数
```

3. 配置MCP服务器路径

修改Financial-MCP-Agent/src/tools/mcp\_config.py

```Python
SERVER_CONFIGS = {
    "a_share_mcp_v2": {  
        "command": "uv", 
        "args": [
            "run",  
            "--directory",
            r"/root/autodl-tmp/Finance/a-share-mcp-is-just-i-need",  # 修改为a-share-mcp-is-just-i-need项目（即MCP服务器）的路径
            "python",  
            "mcp_server.py"  # MCP服务器脚本
        ],
        "transport": "stdio",
    }
}
```

4. lora微调得到**风险分析模型**和**情感分析模型**

在本步骤中，需要使用大语言模型（默认Qwen3-8B）和指定的数据集来训练两个模型：

* 风险分析模型（测试代码在test\_qwen\_risk.py）
* 情感分析模型（测试代码在test\_qwen\_sentiment.py）

这两个模型都将在后面的**新闻分析Agent**中使用。

![](https://v0gbzn36pm.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDM5MzRhNTFhZWEyMjVmZmU4Yzk3ZGZjOGI1ODc2YjZfWDFpeVh4d3pNYmY0dTRGOUpxSjRrZzJIdFJYMjdGQUFfVG9rZW46QThMWWJvaGhPb084Qjh4WTZJSGMwSEwybkZiXzE3NjI5NTMxMTQ6MTc2Mjk1NjcxNF9WNA)

在训练前，需要修改以下路径：

* Qwen模型路径：设置为你本地或远程环境中存放Qwen模型的路径。
* 训练数据集路径：设置为下载好的训练数据集路径。

数据集可以直接使用 HuggingFace 上的两个现成的：

* 风险分析数据集：https://huggingface.co/datasets/benstaf/risk\_nasdaq
* 情感分析数据集：https://huggingface.co/datasets/benstaf/nasdaq\_news\_sentiment

如果你没有可用的 GPU 资源，或者暂时不想进行模型训练，可以在文件`/a-share-mcp-is-just-i-need/src/baostock_data_source.py`中注释掉与风险分析和情感分析相关的代码。这样程序在运行时会跳过这两个模块。

```Plain
cd ..
python train_qwen_risk.py
python train_qwen_sentiment.py
```

5. 测试mcp工具功能

修改`/a-share-mcp-is-just-i-need/src/baostock_data_source.py` 中**情感分析模型**和**风险分析模型**的路径。

（正常情况所有测试都能通过，偶尔会因为网络问题部分测试无法通过）

```Plain
cd a-share-mcp-is-just-i-need
pip install baostock
python test_baostock.py
```

![](https://v0gbzn36pm.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGRmMTNhNjIyYzFkYzI1MjhiNGNhMzBkYWEwOTA4NjZfajZPOTQxSVk1Z3g2dDZUT1ZFNVpvNkpYdVEzeTY3MTJfVG9rZW46TTFpY2J1cmxUbzlwOE94Q2tqSWNtbXJCbkVmXzE3NjI5NTMxMTQ6MTc2Mjk1NjcxNF9WNA)

6. 测试agent功能

用户query参考详见`/Financial-MCP-Agent/test_extraction.py`。生成的markdown报告在Financial-MCP-Agent/reports中。

```Plain
cd ../Financial-MCP-Agent
python src/main.py --command "帮我看看茅台(600519)这只股票值得投资吗"
```

# MCP 讲解

MCP的原理详细见[居丽叶LLM体系知识搭建](https://fp9qo5yj6d.feishu.cn/docx/AN61dRfiWoRUiRxhc6ucbmJwnGr#share-Vlo1dnhGoopHNqx8embcADOtnEc)。简单来说，MCP 是一种让 LLM 使用外部工具的协议：MCP Server 作为工具与资源的提供方，负责注册和提供功能（如本项目中的股票分析、宏观经济数据、指数分析等工具），并同时管理客户端连接、能力协商以及日志与错误报告；MCP Client 则是 LLM 侧的接入者，能够连接到 MCP Server、调用其工具，并将响应格式化后交由 LLM 使用。

## MCP client

* **main.py**​：先从用户提问中提取公司名、股票代码等metadata，然后构建LangGraph工作流，包含5个Agent：基本面分析Agent、技术分析Agent、估值分析Agent、新闻分析Agent和总结Agent，其中前四个Agent可以并行执行。
  
  * **基本面分析Agent**​：根据公司的财务数据，分析公司的盈利、成长空间、运行效率、负债情况等指标。
  * **技术分析Agent**​：根据股票基本信息、最新价格和K线数据，分析价值趋势、成交量和其他技术指标。
  * **估值分析Agent**​：根据公司基本信息和估值分析指标，与业内平均水平进行对比，分析估值变化趋势和股息数据，提供投资建议。
  * **新闻分析Agent**​：根据公司相关的新闻，分析新闻对公司的风险评估和情感。
  * **总结Agent**​：负责汇总前四个Agent的分析结果并生成markdown格式的报告，包括摘要、公司概况、基本面分析、技术分析、估值分析、新闻分析、综合评估、风险因素、投资建议、附录 10 部分。
    前四个Agent的关系可以理解为：
  * **基本面分析Agent**​：关注的是公司的长期发展，如赚钱能力强不强，发展前景如何，供长期投资参考。
  * **技术分析Agent**​：关注短期指标，如最近的涨跌情况，现在买入是否合适，什么时候卖出，供短期投资参考。
  * **估值分析Agent**​：关注价格是否合理，进行横向对比和股份分析（例如市盈率，股价是盈利的多少倍，这个值越大你能赚到的就越少，比如你有100块，A股票100块一股，每股盈利10块；B股票10块一股，每股盈利2元，只考虑市盈率的情况下你应该买B股票）。
  * **新闻分析Agent**​：关注公司最近的新闻，从新闻中分析市场、政策等因素对公司的影响，分析新闻对公司的态度已经预测可能的风险。
  
  用购买手机举例，基本面分析评价手机性能如何；技术分析评价手机销量，价格波动如何；估值分析评价这个手机是否值这个价，跟其他品牌手机对比划不划算；新闻分析评价手机相关的新闻，比如新政策对手机的影响、新品发布、供应链变化等，比如国补政策能更便宜的买到手机，苹果出17后16就会降价。

**This content is only supported in a Feishu Docs**

![](https://v0gbzn36pm.feishu.cn/space/api/box/stream/download/asynccode/?code=YjA5ZjcwNzIwNDA5MmVhN2RkM2ZkZWYxYWUwZGZhY2VfZHpKRXZNTmlHVlEwRzRLVnRSNWlObWVZSWlLT1VFS01fVG9rZW46QXd3SmJyV0tmb2RZSFV4czBmRGMxanp5bnJlXzE3NjI5NTMxMTQ6MTc2Mjk1NjcxNF9WNA)

* **mcp\_client.py**​：定义了**mcp客户端**和​**一些工具函数**​。其中，`print_tool_details`打印可用工具，`get_mcp_tools`负责初始化MCP客户端并获取可用的金融分析工具列表，`close_mcp_client_sessions`负责关闭MCP客户端会话并清理资源，`_main_test_mcp_client`提供了简单的测试接口。

![](https://v0gbzn36pm.feishu.cn/space/api/box/stream/download/asynccode/?code=OGNkMjRlOWY1NTRlOGUzNjZlMjIyOTUwNGU1ZTE4OThfWG13ZUdKWHVvOVRWb2lVVERZOTRmWWFWSFZOQTlxSFJfVG9rZW46TWo1cWJDb3R4b0FPaW94Zm9DS2N4Q0hJbmFkXzE3NjI5NTMxMTQ6MTc2Mjk1NjcxNF9WNA)

* **fundamental\_agent.py**​：实现了一个基于ReAct (Reasoning + Acting)**​ ​**框架的股票​**基本面分析 Agent**​。包含以下几步：
  * 加载模型
  * 调用`get_mcp_tools`获取mcp工具列表（用于获取财务数据、公司信息等）
  * 使用 `langgraph.prebuilt.create_react_agent `创建ReAct Agent，将模型与工具绑定。React原理见[居丽叶LLM体系知识搭建](https://fp9qo5yj6d.feishu.cn/docx/AN61dRfiWoRUiRxhc6ucbmJwnGr#share-KD3hdIhrIobN0JxuoIYcjZUunmd)
  * 构建详细的分析Prompt，包含公司名称、股票代码、当前日期等上下文信息，Prompt明确指定了8个基本面分析维度（公司信息、财务报表、盈利能力、成长能力等）
  * 调用ReAct Agent执行分析，Agent会根据需要自主调用MCP工具获取实际数据
  * 从Agent响应中提取最终分析结果（通常是最后一条AI消息）
  * 记录LLM交互数据（输入、输出、执行时间等），用于后续优化
  * 更新状态，保存分析结果和元数据
  * 添加消息记录，维护对话历史

其他三个agent与基本面分析 Agent的原理类似，区别只体现在提示词，因此不做赘述。

![](https://v0gbzn36pm.feishu.cn/space/api/box/stream/download/asynccode/?code=YjEwMDFlNzE3YWI5MmE3MGFiOGRiMmMzYjVjMGVhYTRfSGhHRVFFQTVwaHRSZE43QlNIUmhZUEFFZUllTUdJb3BfVG9rZW46TUExemJUV0tNb0VWcUN4MEt3eGM5cVFpbkVjXzE3NjI5NTMxMTQ6MTc2Mjk1NjcxNF9WNA)

* **​summary\_agent.py：​**将前面四个 Agent的分析结果整合成最终报告。
  支持两种报告生成方式：
  
  * 本地模型 ：使用 FinR1 模型进行本地推理
  * API 模型 ：通过 OpenAI 兼容接口调用远程大模型
    具体步骤如下：
  * 从状态中获取之前Agent的分析结果和当前时间，并填充入prompt中。
  * 调用大模型生成报告。
  * 记录LLM交互，用于后续分析和优化
  * 将报告存储为markdown格式文件。

![](https://v0gbzn36pm.feishu.cn/space/api/box/stream/download/asynccode/?code=MTY4ZmFlZjUyOGQwYjMzNjNiYzI1NDk3YjUzNTJmYTJfTGVzeDVQbDFvOEMwUG42YWtNNEplN2ZKMjM0UTFaV1FfVG9rZW46WVVzdGJSR0FOb3F2eHR4ZWliY2N2cDhvbk5lXzE3NjI5NTMxMTQ6MTc2Mjk1NjcxNF9WNA)

## MCP server

* **mcp\_server.py**​：首先在服务器端注册了多个MCP工具。

![](https://v0gbzn36pm.feishu.cn/space/api/box/stream/download/asynccode/?code=ZmYxMjE2YWU2M2RlZWIyNjlhY2ZjYmVjMTM0ODc5MTBfTUJ1eWxhV1piSm1xMVphc2FEY0l4OUptTFVrbkFWNlJfVG9rZW46Wndva2JBOGRob1FsOGZ4dllMa2NIV1N1blJnXzE3NjI5NTMxMTQ6MTc2Mjk1NjcxNF9WNA)

* **​stock\_market.py：​**股票市场工具函数总结：

1. `get_historical_k_data`： 历史K线数据获取函数
   1. 功能：获取股票的历史K线数据，支持不同频率和复权方式
   2. 参数：股票代码、日期范围、频率、复权标志、字段列表
   3. 返回：Markdown格式的K线数据表格
2. `get_stock_basic_info` ：股票基本信息获取函数
   1. 功能：获取股票的基本信息，如股票名称、行业、上市日期等
   2. 参数：股票代码、可选字段列表
   3. 返回：Markdown格式的基本信息表格
3. `get_dividend_data`： 分红数据获取函数
   1. 功能：获取股票的分红派息信息
   2. 参数：股票代码、年份、年份类型（预案公告年份或除权除息年份）
   3. 返回：Markdown格式的分红数据表格
4. `get_adjust_factor_data` ：复权因子数据获取函数
   1. 功能：获取股票的复权因子数据，用于计算复权价格
   2. 参数：股票代码、日期范围
   3. 返回：Markdown格式的复权因子数据表格

* **financial\_reports.py**​：财务报表工具函数总结：

1. `get_profit_data `： 盈利能力数据获取函数
   1. 功能：获取股票的季度盈利能力数据（ROE、净利润率等）
   2. 参数：股票代码、年份、季度
   3. 返回：Markdown格式的盈利能力数据表格
2. `get_operation_data` ： 营运能力数据获取函数
   1. 功能：获取股票的季度营运能力数据（周转率等）
   2. 参数：股票代码、年份、季度
   3. 返回：Markdown格式的营运能力数据表格
3. `get_growth_data`： 成长能力数据获取函数
   1. 功能：获取股票的季度成长能力数据（同比增长率等）
   2. 参数：股票代码、年份、季度
   3. 返回：Markdown格式的成长能力数据表格
4. `get_balance_data` ：资产负债表数据获取函数
   1. 功能：获取股票的季度资产负债表/偿债能力数据
   2. 参数：股票代码、年份、季度
   3. 返回：Markdown格式的资产负债表数据表格
5. `get_cash_flow_data` ：现金流量数据获取函数
   1. 功能：获取股票的季度现金流量数据
   2. 参数：股票代码、年份、季度
   3. 返回：Markdown格式的现金流量数据表格
6. `get_dupont_data` ： 杜邦分析数据获取函数
   1. 功能：获取股票的季度杜邦分析数据（ROE分解）
   2. 参数：股票代码、年份、季度
   3. 返回：Markdown格式的杜邦分析数据表格
7. `get_performance_express_report `： 业绩快报数据获取函数
   1. 功能：获取股票的业绩快报数据
   2. 参数：股票代码、开始日期、结束日期
   3. 返回：Markdown格式的业绩快报数据表格
8. `get_forecast_report` ： 业绩预告数据获取函数
   1. 功能：获取股票的业绩预告数据
   2. 参数：股票代码、开始日期、结束日期
   3. 返回：Markdown格式的业绩预告数据表格

* **indices.py**​：指数工具函数总结：

1. `get_stock_industry` ：股票行业分类数据获取函数
   1. 功能：获取指定股票或所有股票的行业分类数据
   2. 参数：股票代码（可选）、日期（可选）
   3. 返回：Markdown格式的行业分类数据表格
2. `get_sz50_stocks`：深证50指数成分股数据获取函数
   1. 功能：获取深证50指数的成分股数据
   2. 参数：日期（可选）
   3. 返回：Markdown格式的深证50指数成分股数据表格
3. `get_hs300_stocks` ：沪深300指数成分股数据获取函数
   1. 功能：获取沪深300指数的成分股数据
   2. 参数：日期（可选）
   3. 返回：Markdown格式的沪深300指数成分股数据表格
4. `get_zz500_stocks` ： 中证500指数成分股数据获取函数
   1. 功能：获取中证500指数的成分股数据
   2. 参数：日期（可选）
   3. 返回：Markdown格式的中证500指数成分股数据表格

* **market\_overview.py**​：市场概览工具函数总结：

1. `get_trade_dates` ： 交易日数据获取函数
   1. 功能：获取指定范围内的交易日信息
   2. 参数：开始日期（可选）、结束日期（可选）
   3. 返回：Markdown格式的交易日数据表格
2. `get_all_stock` ：所有股票数据获取函数
   1. 功能：获取指定日期的所有股票列表及其交易状态
   2. 参数：日期（可选）
   3. 返回：Markdown格式的所有股票数据表格

* **macroeconomic.py**​：宏观经济工具函数总结：

1. `get_deposit_rate_data` ： 存款利率数据获取函数
   1. 功能：获取基准存款利率数据（活期、定期）
   2. 参数：开始日期、结束日期
   3. 返回：Markdown格式的存款利率数据表格
2. `get_loan_rate_data` ： 贷款利率数据获取函数
   1. 功能：获取基准贷款利率数据
   2. 参数：开始日期、结束日期
   3. 返回：Markdown格式的贷款利率数据表格
3. `get_required_reserve_ratio_data` ：存款准备金率数据获取函数
   1. 功能：获取存款准备金率数据
   2. 参数：开始日期、结束日期、年份类型
   3. 返回：Markdown格式的存款准备金率数据表格
4. `get_money_supply_data_month` ：月度货币供应量数据获取函数
   1. 功能：获取月度货币供应量数据（M0、M1、M2）
   2. 参数：开始日期、结束日期
   3. 返回：Markdown格式的月度货币供应量数据表格
5. `get_money_supply_data_year` ： 年度货币供应量数据获取函数
   1. 功能：获取年度货币供应量数据（M0、M1、M2年末余额）
   2. 参数：开始日期、结束日期
   3. 返回：Markdown格式的年度货币供应量数据表格

* **date\_utils.py**​：日期工具函数总结：

1. `get_latest_trading_date`： 最新交易日获取函数
   1. 功能：获取最近的交易日期，如果当天是交易日则返回当天，否则返回最近的交易日
   2. 参数：无
   3. 返回：最近的交易日期，格式为'YYYY-MM-DD'
2. `get_market_analysis_timeframe`： 市场分析时间范围获取函数
   1. 功能：获取适合市场分析的时间范围，基于当前真实日期
   2. 参数：时间范围类型（recent/quarter/half\_year/year）
   3. 返回：包含分析时间范围的详细描述字符串

* **​analysis.py：​**分析工具函数总结：

1. `get_stock_analysis `： 股票分析报告生成函数
   1. 功能：提供基于数据的股票分析报告，而非投资建议
   2. 参数：股票代码、分析类型（fundamental/technical/comprehensive）
   3. 返回：数据驱动的分析报告，包含关键财务指标、历史表现和同行业比较

* **news\_crawler.py**​：新闻分析函数总结

1. `crawl_news`：爬取相关新闻并进行分析
   1. 功能：使用百度搜索爬取与查询词相关的新闻文章，并返回格式化的结果
   2. 参数：搜索查询词、返回新闻数量（默认10条）
   3. 返回：格式化的新闻结果字符串，包含标题、内容摘要、链接等信息

![](https://v0gbzn36pm.feishu.cn/space/api/box/stream/download/asynccode/?code=MDI0YTNmZWVkMTZmY2E5MmNmM2Q0ZjZlOWVlNGRjNzhfWEd2RXNhWDBwNUhpcktSYnJxRXp3ZzN2ZFFtZVJKc2VfVG9rZW46V1F6VWJjbUdGb3V6TVZ4eGZLMmNuc1hUbnJnXzE3NjI5NTMxMTQ6MTc2Mjk1NjcxNF9WNA)

# 工具函数简介

* **baostock\_data\_source.py & utils.py**
  
  * `baostock_login_context`：上下文管理器，处理Baostock登录和登出。
  * `fetch_financial_data`：通用的**财务数据**获取函数，可处理盈利能力、运营能力、成长能力、偿债能力、现金流、杜邦分析等多种财务数据。这些数据都是以季度为周期计算的，衡量的是一个公司运营情况。
    
    * 盈利能力`get_profit_data`：公司赚钱能力，衡量公司一年能赚多少钱？
    * 运营能力`get_operation_data`：公司经营效率，衡量公司的资产利用得好不好，存货周转快不快？
    * 成长能力`get_growth_data`：公司发展前景，衡量公司规模在扩大吗？收入在增长吗？
    * 偿债能力`get_balance_data`： 公司还债能力，衡量公司欠的钱多不多？有没有能力还债？
    * 现金流`get_cash_flow_data`：公司资金流动，衡量现金收支情况，公司实际收到多少现金？花了多少现金？
    * 杜邦分析`get_dupont_data`： 公司赚钱秘诀分析，综合衡量公司赚钱效率、资产利用和财务杠杆。
  * `fetch_index_constituent_data`：**指数成分股数据**获取辅助函数，可处理上证50、沪深300、中证500等各种指数的成分股数据。这些数据就是每天的股市走势，代表整个市场或某个行业。
    
    * 上证50`get_sz50_stocks` = 上海证券交易所最大的50家公司（top-50）
    * 沪深300`get_hs300_stocks` = 沪深两市最大的300家公司（top-300）
    * 中证500`get_zz500_stocks`= 中等规模的500家公司（中位数）
  * `fetch_macro_data`：**宏观经济数据**获取辅助函数，可处理存款利率、贷款利率、存款准备金率、货币供应量、SHIBOR等各种宏观经济数据。这些数据有起始时间，反应整个国家的经济政策和环境怎么样。
    
    * 存款利率/贷款利率`get_deposit_rate_data`​`get_loan_rate_data` ：银行利率，反映银行鼓励你存钱还是借钱。
    * 存款准备金率`get_required_reserve_ratio_data`​`get_money_supply_data_year`：央行的"保证金"，用于控制市场上的钱多少。
    * 货币供应量`get_money_supply_data_month`：市场上有多少钱，反映现金流情况。
    * `get_trade_dates`：获取指定时间范围内的交易日历数据。
      以上三者的关系可以理解为：
    
    1. 财务数据 → 选股票：哪家公司值得投资？
    2. 指数数据 → 看大盘：整个市场表现如何？
    3. 宏观数据 → 判断环境：经济政策对投资有什么影响？
    
    类似于房地产市场的房子质量（财务）、小区环境（指数）、城市发展（宏观）。
  * `fetch_generic_data`：其他数据获取函数。
    
    * `get_historical_k_data`：获取股票历史K线数据，反映（月/周/天）股票价格走势。
    * `get_stock_basic_info`：获取股票基本信息（如股票名称、交易状态等）。
    * `get_dividend_data`：获取股票分红派息数据，公司给股东定期分红作为投资的回报。
    * `get_adjust_factor_data`：获取股票复权因子数据，股票价格会因为分红、送股等原因"跳跃"，复权就是把这种跳跃抹平，让历史价格看起来连续。
    * `get_performance_express_report`：获取股票业绩快报数据，正式财报发布前的一个简化版财报，包括营业收入、净利润、每股收益、同比增长等。
    * `get_forecast_report`： 股票业绩预告数据，基于当前经营情况，预估未来表现，包含预期净利润、增长幅度、分析业绩变动等。
    * `get_stock_industry`： 股票行业分类数据，获取股票所属的行业，用于分析不同行业的整体表现。
    * `get_all_stock`：获取指定日期的全市场股票列表。
  * `crawl_news`：新闻爬取和分析函数，并使用大模型（默认用Qwen3-8B）对新闻进行情感和风险分析，返回格式化的新闻分析结果。
    
    * `_get_article_content`：从给定的URL中提取文章的完整内容，过滤掉过短的文本
    * `_load_risk_model`​`_load_sentiment_model`：加载风险评估和情感分析模型。
    * `_analyze_risk` ：使用风险模型分析新闻内容的风险等级，得分越低表示风险越低。
    * `_analyze_sentiment`：使用情感模型分析新闻内容的情感倾向，得分越低表示情感越负面。
      怎么理解**风险分析**和​**情感分析**​？
  
  1. **风险分析**​：衡量不确定性和潜在损失
  
  * 关注点: 事件的不确定性程度、可能带来的负面影响
  * **时间维度**: 未来可能发生的不利情况
  
  2. 情感分析：衡量市场情绪和短期影响
  
  * 关注点: 市场对新闻的即时反应和情绪倾向
  * 时间维度: 当前的市场情绪状态
  
  举几个例子
  
  1. 新闻: "某AI概念股连续5个涨停，公司宣布进军人工智能领域，股价暴涨200%"
     1. 风险: 5 (极高风险) ， 概念炒作，缺乏实际业绩支撑，存在大幅回调风险
     2. 情感: 5 (极正面) ， 市场极度乐观，追涨情绪浓厚
  2. 新闻: "国家发布新能源汽车补贴政策，补贴金额增加50%，相关产业链受益"
     1. 风险: 2 (低风险) ， 政策确定性高，政府支持力度大
     2. 情感: 4 (正面) ， 政策利好，市场情绪积极
  3. 新闻: "某公司涉嫌财务造假，证监会立案调查，股价跌停，可能面临退市风险"
     1. 风险: 5 (极高风险) ， 财务造假是最严重的风险，可能退市
     2. 情感: 1 (极负面) ， 市场极度恐慌，抛售情绪浓厚
  4. 新闻: "公司某高管出车祸意外死亡，但公司经营正常，业绩稳定"
     1. 风险: 3(中等风险) ， 个人问题不影响公司基本面
     2. 情感: 2 (轻微负面) ， 负面新闻影响市场情绪

![](https://v0gbzn36pm.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDI2ZDJjYTc1NjJmMzczZDdiMjg3YzMxYzI3ZmNjNWNfc0RyMkpoQVdnNXE2cGFTa1BtUXVjY1N6VGtHOXlsOThfVG9rZW46QnpXd2J5Sm1Eb3liQnN4bnBUbWNhdk9SbmRaXzE3NjI5NTMxMTQ6MTc2Mjk1NjcxNF9WNA)

参考项目：

* https://github.com/SUFE-AIFLM-Lab/Fin-R1/tree/main
* https://github.com/24mlight/Financial-MCP-Agent/tree/main
* https://github.com/24mlight/a-share-mcp-is-just-i-need/tree/main
* https://github.com/AI4Finance-Foundation/FinRL?tab=readme-ov-file

# 项目相关的股票基础知识

1. **基本面**

**目的**​：弄清楚“这家公司是否值得投资”，研究企业的经营情况和长期成长潜力。

**宏观层面**

* **经济指标**​：GDP增速、PMI、工业增加值、消费数据
* **货币政策**​：利率、存款准备金率、货币供应量（M2）
* **财政政策**​：基建投资、专项债发行
* **国际环境**​：美联储加息/降息、汇率波动、国际大宗商品价格
* **A股特点**​：政策市色彩浓厚，监管政策、产业扶持政策影响巨大

**行业层面**

* **周期性行业**​：如钢铁、煤炭、化工，受经济波动影响大
* **成长性行业**​：如新能源、半导体、人工智能
* **防御性行业**​：医药、公用事业、消费必需品
* **行业地位**​：行业集中度（CR值）、龙头企业的护城河

**公司层面**

* **财务分析**​：
  * **盈利能力**​：ROE、毛利率、净利率
  * **成长性**​：营收、净利润增速
  * **偿债能力**​：资产负债率、流动比率
  * **现金流**​：经营性现金流净额
* **竞争优势**​：技术壁垒、品牌、渠道、规模效应
* **管理层与股东结构**​：是否有大股东减持、是否存在利益输送问题
* **企业生命周期**​：初创期、成长期、成熟期、衰退期
* **定性因素​**​：公司治理、管理层能力、核心竞争力、护城河

2. **技术**

**目的**​：研究“什么时候买卖”，主要用于短期和中期的交易决策

**价格（Price）**​：价格是市场供需的直接反映

* **趋势特征**
  * **上涨趋势**​：不断创出“更高的高点”和“更高的低点”
  * **下跌趋势**​：不断创出“更低的低点”和“更低的高点”
  * **震荡趋势**​：价格在区间内上下波动，没有明显方向
* **K线信息**
  * **实体**​：开盘价与收盘价的差值，代表买卖力量对比
  * **上下影线**​：说明多空博弈，影线长意味着争夺激烈
  * **组合形态**​：如大阳线（强势）、十字星（犹豫）、锤子线（可能反转）

**成交量（Volume）**​：成交量是市场参与资金的体现，常被视为“价格的发动机”

* **量能大小**​：放量代表交易活跃，缩量代表观望气氛浓
* **量价关系**​：价格走势是否有成交量配合，是判断行情是否有效的关

**技术指标**

* **均线系统（MA）**​：短期（5、10日）、中期（20、60日）、长期（120、250日）
* **MACD**​：红绿柱变化，判断趋势强弱与反转信号
* **KDJ / RSI**​：超买超卖指标，常用于短线
* **BOLL布林带**​：股价波动范围判断

**形态学**

* **头肩顶/底**​：反转形态
* **双顶/双底**​：顶部和底部确认
* **旗形、三角形**​：整理形态，常伴随突破行情

**量价关系**

* **价升量增**​：趋势确认
* **价升量缩**​：上涨动能不足
* **量增价跌**​：下跌加速
* **量缩价稳**​：底部可能形成

> **A股特色**​：散户占比大，短期动量（涨的股票更容易涨），中期反转（涨的股票更容易跌）

3. **估值**

**目的**​：研究“股价是贵还是便宜”，避免高位接盘或错过低估机会

**常见估值指标**

* **PE（市盈率）**​：价格 ÷ 每股收益（EPS），适合盈利稳定企业
* **PB（市净率）**​：价格 ÷ 每股净资产，适合金融股、周期股
* **PS（市销率）**​：市值 ÷ 营收，适合初创或高成长公司
* **EV/EBITDA**​：常用国际指标，排除资本结构差异
* **PEG**​：PE ÷ 增长率，兼顾估值与成长性

**估值方法**

* **相对估值**​：与同行业可比公司比较（横向）
* **历史估值对比**​：与自身历史估值比较（纵向）
* **绝对估值**​：DCF（现金流折现）、股利折现模型

**A股市场的估值特点**

* 成长股往往有“估值溢价”。
* 政策导向行业容易被资金追捧，估值快速上升
* 周期股估值容易在高低之间来回波动

4. **新闻与信息面**

**目的**​：研究“短期刺激因素”，往往决定股价的阶段性波动

**政策消息**

* **宏观政策**​：货币政策、财政政策
* **产业政策**​：补贴、税收优惠（新能源、芯片）
* **监管政策**​：IPO节奏、减持新规、房地产调控

**公司消息**

* **业绩预告/快报**​：超预期利好/利空
* **重大合同、并购重组**​：股价催化剂
* **股权变动**​：大股东增持/减持
* **分红送股**​：现金分红、送转股

**市场舆情**

* **媒体报道**​：热点题材传播
* **券商研报**​：对个股或行业的评级变化
* **社交舆情**​：散户炒作氛围（雪球、微博等）

**突发事件**

* **国际因素​**​：地缘冲突、海外金融风险
* **突发利空**​：黑天鹅事件、财务造假
* **自然灾害/疫情**​：对产业链和消费的影响

**A股特点**​：消息驱动效应强，题材炒作明显，经常出现“消息一出，股价大涨/大跌”的情况

