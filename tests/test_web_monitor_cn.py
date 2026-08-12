from pathlib import Path


MONITOR_HTML = Path(__file__).resolve().parents[1] / "web" / "static" / "monitor.html"


def test_monitor_core_labels_are_chinese():
    html = MONITOR_HTML.read_text(encoding="utf-8")

    expected = [
        "量子任务控制台",
        "任务触发器",
        "标准运行",
        "注入幻觉",
        "漂移事件",
        "帕累托扫描",
        "实时模式",
        "实时 AI 监看",
        "遥测来源",
        "系统信息",
        "实时 ReAct 轨迹",
        "Verifier 判定",
        "智能体最终回答",
        "漂移后保真度",
        "工具调用分布",
        "量子后端状态",
        "事件日志",
        "选择 LIVE 演示任务",
    ]

    for label in expected:
        assert label in html
