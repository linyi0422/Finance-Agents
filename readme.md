# Finance-Agents | 金融分析多智能体系统

一句话：基于 LangGraph + MCP 的 A 股投研分析智能体，自动汇总财务 / 技术 / 估值 / 新闻并生成结构化报告。

**定位**
- 使用场景：对单只股票做快速投研梳理与要点总结
- 输出形态：Markdown 报告（摘要、基本面、技术面、估值、新闻、风险、建议）
- 运行方式：支持 API 大模型；总结 Agent 可切换本地 FinR1

**亮点**
- 5 个专业 Agent 并行分析，最后统一汇总
- MCP 工具统一接入财务、行情、指数、宏观与新闻数据
- 报告结构化、可读性高，适合展示与二次加工
- 训练脚本支持风险 / 情感 LoRA 微调
- 代码组织清晰，可拓展新的 Agent 或工具

**演示**
- 报告示例：`docs/report_sample_600519_20260131.md`（精选展示版）
- 原始报告：`reports/` 与 `Financial-MCP-Agent/reports/`
- 交互入口：`Financial-MCP-Agent/src/main.py`
- Notebook 示例：`demo_sentiment_usage.ipynb`，`demo_sentiment_usage_min.ipynb`

**报告摘录（样例）**
- 执行摘要：“短期呈下降趋势，长期战略方向明确；但基本面与估值数据需后续财报验证。”
- 风险等级：“中等（改革执行、宏观影响、技术面破位）”
- 预期回报：“长期谨慎乐观”
- 技术面：“关注 9,900-10,000 元支撑区；若跌破需防范进一步下探。”
- 投资建议：“综合评级：持有 / 观望，适合长期价值投资者分批关注。”
说明：完整报告见 `docs/report_sample_600519_20260131.md`。

**架构概览**
```text
User Query
  -> LangGraph Orchestrator
    -> Fundamental Agent
    -> Technical Agent
    -> Value Agent
    -> News Agent
    -> Summary Agent (aggregate)
  -> Markdown Report
```

**核心模块**
- `Financial-MCP-Agent/src/main.py` 入口与工作流编排
- `Financial-MCP-Agent/src/agents/*.py` 五类 Agent
- `Financial-MCP-Agent/src/tools/mcp_client.py` MCP 客户端封装
- `Financial-MCP-Agent/src/tools/mcp_config.py` MCP 服务配置

**技术栈**
- LangGraph, LangChain, MCP
- Python, Transformers, PEFT
- OpenAI 兼容接口 / 本地 FinR1
- Baostock 与多源金融数据

**快速运行（展示用最小步骤）**
```bash
pip install -r requirements.txt
cp Financial-MCP-Agent/.env.example Financial-MCP-Agent/.env
python Financial-MCP-Agent/src/main.py --command "帮我看看茅台(600519)这只股票值不值得投资"
```
说明：详细部署与训练说明见 `docs/README_FULL.md`。

**项目亮点**
- 设计多 Agent 分工与汇总报告结构
- 整合 MCP 工具与多源金融数据
- 训练 / 接入风险与情感模型（LoRA）
- 输出可复用的报告模板与演示 Notebook

**备注**
- 本仓库为展示用途，引用了若干开源组件与数据集，详见 `docs/README_FULL.md`。
