import io
import math
from importlib.util import find_spec
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
except ImportError:
    pdk = None


st.set_page_config(
    page_title="城市綠網空品植生分析工具",
    page_icon="🌿",
    layout="wide",
)


# ============================================================
# 一、欄位設定
# ============================================================

REQUIRED_COLUMNS = [
    "代碼",
    "縣市",
    "鄉鎮市區",
    "基地名稱",
    "基地類型(校正)",
    "TWD97X",
    "TWD97Y",

    # 節點潛力
    "500公尺內工廠數",
    "距主要道路距離(公尺)",
    "500公尺內敏感受體數",
    "鄉鎮市區人口密度(人/平方公里)",
    "鄉鎮市區年均溫(°C)",
    "土地權屬",
    "管理機關",
    "基地面積(公頃)",
    "基地長度(公里)",
    "短期推動性",
    "開放可及性",
    "500公尺內生活節點數",

    # 串聯潛力
    "最近綠化單元距離(公尺)",
    "500公尺內其他綠化單元數",
    "1000公尺內其他綠化單元數",
    "1000公尺內其他綠化單元總面積(公頃)",

    # 功能角色判釋
    "基地內部是否有停留活動空間",
    "是否有步行通學騎行路徑",
    "是否有邊界防護需求",
    "道路周邊是否有敏感受體或社區活動空間",
    "是否有短期事件",
    "短期事件是否影響內部空間",
    "短期事件是否影響通行路徑",
    "短期事件是否位於邊界或鄰近受體",
]

NUMERIC_COLUMNS = [
    "TWD97X",
    "TWD97Y",
    "500公尺內工廠數",
    "距主要道路距離(公尺)",
    "500公尺內敏感受體數",
    "鄉鎮市區人口密度(人/平方公里)",
    "鄉鎮市區年均溫(°C)",
    "基地面積(公頃)",
    "基地長度(公里)",
    "500公尺內生活節點數",
    "最近綠化單元距離(公尺)",
    "500公尺內其他綠化單元數",
    "1000公尺內其他綠化單元數",
    "1000公尺內其他綠化單元總面積(公頃)",
]

YES_NO_COLUMNS = [
    "基地內部是否有停留活動空間",
    "是否有步行通學騎行路徑",
    "是否有邊界防護需求",
    "道路周邊是否有敏感受體或社區活動空間",
    "是否有短期事件",
    "短期事件是否影響內部空間",
    "短期事件是否影響通行路徑",
    "短期事件是否位於邊界或鄰近受體",
]

NODE_COLORS: Dict[str, List[int]] = {
    "高潛力": [0, 120, 60, 190],
    "中潛力": [255, 170, 0, 180],
    "基礎潛力": [160, 160, 160, 160],
}

LINK_COLORS: Dict[str, List[int]] = {
    "高潛力": [0, 90, 180, 190],
    "中潛力": [255, 170, 0, 180],
    "基礎潛力": [160, 160, 160, 160],
}

ROLE_COLORS: Dict[str, List[int]] = {
    "A1": [46, 204, 113, 180],
    "A2": [26, 188, 156, 180],
    "A3": [39, 174, 96, 180],
    "B1": [52, 152, 219, 180],
    "B2": [41, 128, 185, 180],
    "B3": [31, 97, 141, 190],
    "C1": [230, 126, 34, 180],
    "C2": [211, 84, 0, 180],
    "C3": [192, 57, 43, 190],
    "未判定": [127, 140, 141, 160],
}


ROLE_INFO = {
    "A1": {
        "情境名稱": "A1 多源逸散背景下之長時間停留暴露情境",
        "主要功能": "攔截與沉降、心理／行為調節",
        "適用性": "可導入",
        "管理方式": "以提升停留空間品質為主，維持植栽覆蓋完整、減少裸露地表，並定期檢查積塵、植栽健康與下層覆蓋情形。若問題主要為區域性 PM2.5 或二次污染，植生僅宜作為補強措施。",
        "追蹤重點": "積塵情形、裸露地變化、葉面與下層植栽狀況、停留空間舒適性。",
    },
    "A2": {
        "情境名稱": "A2 多源逸散背景下之短時間移動暴露情境",
        "主要功能": "擾流與稀釋、心理／行為調節",
        "適用性": "可導入",
        "管理方式": "以降低通行者接觸高暴露點的時間與頻率為主，優先檢討動線調整、遮蔭、帶狀綠化與路徑導引。",
        "追蹤重點": "通行舒適性、路徑使用情形、是否仍有高暴露停留點。",
    },
    "A3": {
        "情境名稱": "A3 多源逸散背景下之邊界暴露情境",
        "主要功能": "阻隔與緩衝、攔截與沉降",
        "適用性": "優先導入",
        "管理方式": "以建立連續、多層次、穩定的邊界植栽帶為主，降低源區污染向受體區直接傳輸。",
        "追蹤重點": "邊界連續性、缺口、植栽密度、下風處粉塵或異味感受。",
    },
    "B1": {
        "情境名稱": "B1 線源擾動背景下之長時間停留暴露情境",
        "主要功能": "阻隔與緩衝、攔截與沉降",
        "適用性": "可導入",
        "管理方式": "以道路與受體之間的邊界防護為主，配置應掌握「可防護而不封閉」原則，避免因過密配置造成污染停滯。",
        "追蹤重點": "植栽是否位於道路與受體之間、通風條件、近道路受體使用情形、粉塵感受變化。",
    },
    "B2": {
        "情境名稱": "B2 線源擾動背景下之短暫通行暴露情境",
        "主要功能": "擾流與稀釋、心理／行為調節",
        "適用性": "可導入",
        "管理方式": "以改善通行舒適性、遮蔭、分隔人流與車流為主，避免讓人流更靠近污染源或阻礙通風。",
        "追蹤重點": "通行動線合理性、植栽是否影響通風、遮蔭與視覺緩衝效果。",
    },
    "B3": {
        "情境名稱": "B3 線源擾動背景下之邊界暴露情境",
        "主要功能": "阻隔與緩衝",
        "適用性": "優先導入",
        "管理方式": "以連續植栽帶降低道路污染直接傳輸，並依道路尺度、受體位置與風向條件調整高度、寬度與孔隙度。",
        "追蹤重點": "邊界完整性、通風條件、受體側粉塵或異味感受、植栽有效防護位置。",
    },
    "C1": {
        "情境名稱": "C1 事件型或作業型污染對長時間停留受體之影響情境",
        "主要功能": "攔截與沉降、阻隔與緩衝",
        "適用性": "補強使用",
        "管理方式": "不建議以植生作為優先管理工具，應優先採取源頭抑制、覆蓋、灑水、工法改善、排放控制及作業管理。植生僅作為視覺遮蔽、局部緩衝或生活圈環境整合的補強措施。",
        "追蹤重點": "事件是否持續、源頭控制是否落實、植生是否被誤用為主要改善手段。",
    },
    "C2": {
        "情境名稱": "C2 事件型污染影響短時間移動暴露之情境",
        "主要功能": "擾流與稀釋、心理／行為調節",
        "適用性": "補強使用",
        "管理方式": "原則上不建議作為植生淨化優先導入類型，應以事件管理、作業時間調整、動線管制、臨時防制設施與源頭控制為優先。植生可輔助遮蔭、視覺緩衝或動線整理。",
        "追蹤重點": "臨時動線是否避開污染熱點、事件結束後是否需調整或移除臨時措施。",
    },
    "C3": {
        "情境名稱": "C3 事件型或間歇型污染源與受體之邊界暴露情境",
        "主要功能": "阻隔與緩衝、心理／行為調節",
        "適用性": "慎用／補強使用",
        "管理方式": "可作為周界防護、視覺遮蔽與感知緩衝，但不可取代源頭污染控制，應與工程防制、作業管理及排放控制併行。",
        "追蹤重點": "邊界完整性、下風處受體感受、陳情頻率、事件高峰期間暴露變化。",
    },
}


# ============================================================
# 二、通用工具函式
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    return data


def check_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def clean_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if value in ["", "-", "—", "無", "NA", "N/A", "未知", "不明"]:
            return np.nan
    return pd.to_numeric(value, errors="coerce")


def normalize_yes_no(value) -> str:
    """將是／否／未知欄位標準化。"""
    if pd.isna(value):
        return "未知"

    text = str(value).strip().lower()
    if text in ["是", "yes", "y", "true", "1", "有", "符合"]:
        return "是"
    if text in ["否", "no", "n", "false", "0", "無", "不符合"]:
        return "否"
    return "未知"


def is_yes(value) -> bool:
    return normalize_yes_no(value) == "是"


def classify_level(score: float, high_cut: float, medium_cut: float) -> str:
    if score >= high_cut:
        return "高潛力"
    if score >= medium_cut:
        return "中潛力"
    return "基礎潛力"


def text_contains_any(value, keywords: List[str]) -> bool:
    text = "" if pd.isna(value) else str(value)
    return any(k in text for k in keywords)


def twd97_to_wgs84(x, y, county="") -> Tuple[float, float]:
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan

    lon0_degree = 119 if str(county).strip() in ["金門縣"] else 121
    a = 6378137.0
    b = 6356752.314245
    lon0 = math.radians(lon0_degree)
    k0 = 0.9999
    dx = 250000.0
    e = math.sqrt(1 - (b / a) ** 2)
    x = float(x) - dx
    y = float(y)
    m = y / k0
    mu = m / (a * (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))
    e1 = (1 - math.sqrt(1 - e**2)) / (1 + math.sqrt(1 - e**2))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)
    e2 = e**2 / (1 - e**2)
    c1 = e2 * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    r1 = a * (1 - e**2) / ((1 - e**2 * math.sin(fp) ** 2) ** 1.5)
    n1 = a / math.sqrt(1 - e**2 * math.sin(fp) ** 2)
    d = x / (n1 * k0)
    q1 = n1 * math.tan(fp) / r1
    q2 = d**2 / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * e2) * d**4 / 24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 3 * c1**2 - 252 * e2) * d**6 / 720
    lat = fp - q1 * (q2 - q3 + q4)
    q5 = d
    q6 = (1 + 2 * t1 + c1) * d**3 / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * e2 + 24 * t1**2) * d**5 / 120
    lon = lon0 + (q5 - q6 + q7) / math.cos(fp)
    return math.degrees(lat), math.degrees(lon)


# ============================================================
# 三、節點潛力評分
# ============================================================

def score_air_pressure(row: pd.Series) -> int:
    factories = row["500公尺內工廠數"]
    road = row["距主要道路距離(公尺)"]

    high = (not pd.isna(factories) and factories >= 3) or (not pd.isna(road) and road <= 100)
    medium = (
        (not pd.isna(factories) and 1 <= factories <= 2)
        or (not pd.isna(road) and 100 < road <= 300)
    )

    if high:
        return 2
    if medium:
        return 1
    return 0


def score_sensitive_receptors(value) -> int:
    if pd.isna(value):
        return 0
    if value >= 3:
        return 2
    if value >= 1:
        return 1
    return 0


def score_population_density(value) -> int:
    if pd.isna(value):
        return 0
    if value >= 1000:
        return 2
    if value >= 500:
        return 1
    return 0


def score_temperature(value) -> int:
    if pd.isna(value):
        return 0
    if value >= 24:
        return 2
    if value >= 22:
        return 1
    return 0


def score_land_ownership(value) -> int:
    text = "" if pd.isna(value) else str(value).strip()

    if text == "":
        return 0

    if any(k in text for k in ["公有", "國有", "市有", "縣有", "鄉有", "鎮有", "區有", "單一權屬"]):
        return 2

    if any(k in text for k in ["公私混合", "混合", "部分公有", "跨機關", "需協調"]):
        return 1

    if any(k in text for k in ["私有", "權屬複雜", "複雜", "徵收"]):
        return 0

    # 不明確但有資料者，暫列為需確認
    return 1


def score_management_agency(value) -> int:
    text = "" if pd.isna(value) else str(value).strip()

    if text == "":
        return 0

    if any(k in text for k in ["無", "不明", "未知", "未定", "待確認"]):
        return 0

    if any(k in text for k in ["需確認", "可能", "暫由", "協調"]):
        return 1

    return 2


def get_effective_plantable_area(row: pd.Series) -> float:
    area = row["基地面積(公頃)"]
    length = row["基地長度(公里)"]

    if not pd.isna(area):
        return float(area)

    # 若只有長度資料，假設寬度 1 公尺：
    # 1 公里 × 1 公尺 = 1000 平方公尺 = 0.1 公頃
    if not pd.isna(length):
        return float(length) * 0.1

    return np.nan


def score_plantable_space(row: pd.Series) -> int:
    area = get_effective_plantable_area(row)

    if pd.isna(area):
        return 0
    if area > 1:
        return 2
    if area > 0.5:
        return 1
    return 0


def score_short_term_feasibility(value) -> int:
    text = "" if pd.isna(value) else str(value).strip()

    if text == "":
        return 0

    if any(k in text for k in ["已納入計畫", "已有預算", "可立即推動", "可短期推動", "高", "是"]):
        return 2

    if any(k in text for k in ["需部分協調", "需協調", "中", "部分", "待評估"]):
        return 1

    if any(k in text for k in ["短期難以推動", "重大協調", "尚無政策支持", "低", "否", "困難"]):
        return 0

    return 1


def score_openness(value) -> int:
    text = "" if pd.isna(value) else str(value).strip()

    if text == "":
        return 0

    if any(k in text for k in ["完全開放", "自由進入", "全日開放", "開放"]):
        if "不開放" not in text and "部分" not in text:
            return 2

    if any(k in text for k in ["部分開放", "特定時段", "特定區域", "需申請", "半開放"]):
        return 1

    if any(k in text for k in ["不開放", "未開放", "封閉", "內部管理"]):
        return 0

    return 1


def score_life_nodes(value) -> int:
    if pd.isna(value):
        return 0
    if value >= 4:
        return 2
    if value >= 2:
        return 1
    return 0


# ============================================================
# 四、串聯潛力評分
# ============================================================

def score_nearest_green(value) -> int:
    if pd.isna(value):
        return 0
    if value <= 100:
        return 2
    if value <= 300:
        return 1
    return 0


def score_green_units_500(value) -> int:
    if pd.isna(value):
        return 0
    if value >= 4:
        return 2
    if value >= 2:
        return 1
    return 0


def score_green_units_1000(value) -> int:
    if pd.isna(value):
        return 0
    if value >= 8:
        return 2
    if value >= 4:
        return 1
    return 0


def score_green_area_1000(value) -> int:
    if pd.isna(value):
        return 0
    if value >= 10:
        return 2
    if value >= 3:
        return 1
    return 0


# ============================================================
# 五、功能角色判釋
# ============================================================

def classify_function_roles(row: pd.Series) -> List[str]:
    roles = []

    q1 = is_yes(row["基地內部是否有停留活動空間"])
    q2 = is_yes(row["是否有步行通學騎行路徑"])
    q3 = is_yes(row["是否有邊界防護需求"])
    q5 = is_yes(row["道路周邊是否有敏感受體或社區活動空間"])
    q6 = is_yes(row["是否有短期事件"])
    q7 = is_yes(row["短期事件是否影響內部空間"])
    q8 = is_yes(row["短期事件是否影響通行路徑"])
    q9 = is_yes(row["短期事件是否位於邊界或鄰近受體"])

    road_distance = row["距主要道路距離(公尺)"]
    near_major_road = not pd.isna(road_distance) and road_distance <= 300

    # A 類：多源逸散或基地常態空間情境
    if q1:
        roles.append("A1")
    if q2:
        roles.append("A2")
    if q3:
        roles.append("A3")

    # B 類：道路或交通線源情境
    if near_major_road and q5:
        roles.append("B1")
    if near_major_road and q2:
        roles.append("B2")
    if near_major_road and q3 and q5:
        roles.append("B3")

    # C 類：短期事件情境
    if q6 and q7:
        roles.append("C1")
    if q6 and q8:
        roles.append("C2")
    if q6 and q9:
        roles.append("C3")

    return roles


def role_codes_to_text(roles: List[str]) -> str:
    return "、".join(roles) if roles else "未判定"


def roles_to_names(roles: List[str]) -> str:
    if not roles:
        return "未判定"
    return "；".join(ROLE_INFO[r]["情境名稱"] for r in roles)


def roles_to_functions(roles: List[str]) -> str:
    if not roles:
        return "未判定"
    funcs = []
    for r in roles:
        funcs.append(f"{r}：{ROLE_INFO[r]['主要功能']}")
    return "；".join(funcs)


def roles_to_applicability(roles: List[str]) -> str:
    if not roles:
        return "未判定"
    return "；".join(f"{r}：{ROLE_INFO[r]['適用性']}" for r in roles)


def roles_to_management(roles: List[str]) -> str:
    if not roles:
        return "尚無明確功能情境，建議補充現地判讀欄位後再確認。"
    return "；".join(f"{r}：{ROLE_INFO[r]['管理方式']}" for r in roles)


def roles_to_tracking(roles: List[str]) -> str:
    if not roles:
        return "建議補充功能情境資料。"
    return "；".join(f"{r}：{ROLE_INFO[r]['追蹤重點']}" for r in roles)


def first_role_code(roles_text: str) -> str:
    if not isinstance(roles_text, str) or roles_text == "未判定":
        return "未判定"
    return roles_text.split("、")[0]



# ============================================================
# 六、優先推動建議
# ============================================================

def build_priority_recommendation(row: pd.Series) -> str:
    """整合節點潛力、串聯潛力與功能情境，產生可回扣條件的精簡建議。"""
    node_level = row.get("節點潛力", "基礎潛力")
    link_level = row.get("串聯潛力", "基礎潛力")
    roles_text = row.get("可能功能情境", "未判定")

    # 主建議：回扣節點潛力與串聯潛力
    if node_level == "高潛力" and link_level == "高潛力":
        main = "節點與串聯條件皆佳，建議優先推動"
    elif node_level == "高潛力":
        main = "節點條件佳，建議優先強化基地功能"
    elif link_level == "高潛力":
        main = "串聯條件佳，建議作為綠網連接補點"
    elif node_level == "中潛力" and link_level == "中潛力":
        main = "節點與串聯條件中等，建議納入第二階段評估"
    elif node_level == "基礎潛力" and link_level == "中潛力":
        main = "串聯條件尚可，建議作為局部連接或補強場址"
    elif node_level == "中潛力" and link_level == "基礎潛力":
        main = "節點條件尚可，建議視管理可行性再評估"
    else:
        main = "節點與串聯條件皆低，建議暫列低優先序"

    # 功能補充：回扣功能情境
    if not isinstance(roles_text, str) or roles_text == "未判定":
        note = "功能情境尚未明確，建議補充現地判讀"
    elif any(code in roles_text for code in ["C1", "C2", "C3"]):
        note = "另具事件型情境，應以源頭或作業管理為優先"
    elif any(code in roles_text for code in ["A3", "B3"]):
        note = "另具邊界情境，應檢討連續植栽緩衝"
    elif any(code in roles_text for code in ["A2", "B2"]):
        note = "另具通行暴露情境，應優化通行動線與遮蔭"
    elif any(code in roles_text for code in ["A1", "B1"]):
        note = "另具停留暴露情境，應強化停留空間防護"
    else:
        note = "功能情境尚未明確，建議補充現地判讀"

    return f"{main}；{note}。"


# ============================================================
# 六、主分析函式
# ============================================================

def analyze_green_network(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    for col in NUMERIC_COLUMNS:
        data[col] = data[col].apply(clean_numeric)

    for col in YES_NO_COLUMNS:
        data[col] = data[col].apply(normalize_yes_no)

    coords = data.apply(lambda r: twd97_to_wgs84(r["TWD97X"], r["TWD97Y"], r["縣市"]), axis=1)
    data["緯度"] = coords.apply(lambda x: x[0])
    data["經度"] = coords.apply(lambda x: x[1])

    # 節點潛力，10 項，每項 0–2 分，滿分 20 分
    data["節點_空品壓力分數"] = data.apply(score_air_pressure, axis=1)
    data["節點_敏感受體分數"] = data["500公尺內敏感受體數"].apply(score_sensitive_receptors)
    data["節點_人口暴露分數"] = data["鄉鎮市區人口密度(人/平方公里)"].apply(score_population_density)
    data["節點_高溫風險分數"] = data["鄉鎮市區年均溫(°C)"].apply(score_temperature)
    data["節點_土地權屬分數"] = data["土地權屬"].apply(score_land_ownership)
    data["節點_管理權責分數"] = data["管理機關"].apply(score_management_agency)
    data["估算可植栽面積(公頃)"] = data.apply(get_effective_plantable_area, axis=1)
    data["節點_可植栽空間分數"] = data.apply(score_plantable_space, axis=1)
    data["節點_短期推動分數"] = data["短期推動性"].apply(score_short_term_feasibility)
    data["節點_開放可及分數"] = data["開放可及性"].apply(score_openness)
    data["節點_生活節點分數"] = data["500公尺內生活節點數"].apply(score_life_nodes)

    node_score_columns = [
        "節點_空品壓力分數",
        "節點_敏感受體分數",
        "節點_人口暴露分數",
        "節點_高溫風險分數",
        "節點_土地權屬分數",
        "節點_管理權責分數",
        "節點_可植栽空間分數",
        "節點_短期推動分數",
        "節點_開放可及分數",
        "節點_生活節點分數",
    ]
    data["節點潛力分數"] = data[node_score_columns].sum(axis=1)
    data["節點潛力"] = data["節點潛力分數"].apply(lambda s: classify_level(s, high_cut=15, medium_cut=8))

    # 串聯潛力，4 項，每項 0–2 分，滿分 8 分
    data["串聯_最近綠化距離分數"] = data["最近綠化單元距離(公尺)"].apply(score_nearest_green)
    data["串聯_500公尺綠化單元分數"] = data["500公尺內其他綠化單元數"].apply(score_green_units_500)
    data["串聯_1000公尺綠化單元分數"] = data["1000公尺內其他綠化單元數"].apply(score_green_units_1000)
    data["串聯_1000公尺綠化面積分數"] = data["1000公尺內其他綠化單元總面積(公頃)"].apply(score_green_area_1000)

    link_score_columns = [
        "串聯_最近綠化距離分數",
        "串聯_500公尺綠化單元分數",
        "串聯_1000公尺綠化單元分數",
        "串聯_1000公尺綠化面積分數",
    ]
    data["串聯潛力分數"] = data[link_score_columns].sum(axis=1)
    data["串聯潛力"] = data["串聯潛力分數"].apply(lambda s: classify_level(s, high_cut=6, medium_cut=3))

    # 功能角色
    role_lists = data.apply(classify_function_roles, axis=1)
    data["可能功能情境"] = role_lists.apply(role_codes_to_text)
    data["功能情境名稱"] = role_lists.apply(roles_to_names)
    data["主要對應功能"] = role_lists.apply(roles_to_functions)
    data["植生措施適用性"] = role_lists.apply(roles_to_applicability)
    data["建議後續管理方式"] = role_lists.apply(roles_to_management)
    data["追蹤重點"] = role_lists.apply(roles_to_tracking)

    data["地圖角色代碼"] = data["可能功能情境"].apply(first_role_code)

    data["優先推動建議"] = data.apply(build_priority_recommendation, axis=1)

    return data


# ============================================================
# 七、讀檔與匯出
# ============================================================

def read_csv_with_fallback(uploaded_file) -> pd.DataFrame:
    """讀取 CSV，依序嘗試常見編碼。"""
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5", "latin1"]
    last_error = None

    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e
        except Exception as e:
            last_error = e

    raise ValueError(f"CSV 讀取失敗，已嘗試編碼：{', '.join(encodings)}。最後錯誤：{last_error}")


def make_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_excel_download(df: pd.DataFrame):
    if find_spec("openpyxl") is None:
        return None

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="分析結果")

        node_rules = pd.DataFrame(
            {
                "模組": ["節點潛力"] * 10,
                "指標": [
                    "空品壓力",
                    "敏感受體密度",
                    "鄉鎮市區人口暴露程度",
                    "鄉鎮市區高溫風險",
                    "土地權屬單純度",
                    "管理權責明確性",
                    "可植栽空間",
                    "短期推動性",
                    "開放可及性",
                    "鄰近生活節點",
                ],
                "評分方式": [
                    "工廠數≥3或距主要道路≤100m為2分；工廠數1–2或距道路101–300m為1分；其餘0分。",
                    "500m內敏感受體≥3為2分；1–2為1分；0為0分。",
                    "鄉鎮人口密度≥1000人/km²為2分；500–999為1分；<500為0分。",
                    "年均溫≥24°C為2分；22–23.9°C為1分；<22°C為0分。",
                    "公有或單一權屬為2分；混合或需協調為1分；私有或複雜為0分。",
                    "管理機關明確為2分；需確認為1分；無明確管理單位為0分。",
                    "面積>1ha為2分；>0.5–1ha為1分；≤0.5ha為0分。若無面積，以長度×0.1估算。",
                    "可短期推動為2分；需部分協調為1分；短期困難為0分。",
                    "完全開放為2分；部分開放為1分；不開放為0分。",
                    "500m內生活節點≥4為2分；2–3為1分；0–1為0分。",
                ],
            }
        )

        link_rules = pd.DataFrame(
            {
                "模組": ["串聯潛力"] * 4,
                "指標": [
                    "最近綠化單元距離",
                    "500公尺內其他綠化單元數",
                    "1000公尺內其他綠化單元數",
                    "1000公尺內其他綠化單元總面積",
                ],
                "評分方式": [
                    "≤100m為2分；>100–300m為1分；>300m為0分。",
                    "≥4處為2分；2–3處為1分；0–1處為0分。",
                    "≥8處為2分；4–7處為1分；0–3處為0分。",
                    "≥10ha為2分；3–9.99ha為1分；<3ha為0分。",
                ],
            }
        )

        role_rules = pd.DataFrame(
            [
                {"功能情境": code, **info}
                for code, info in ROLE_INFO.items()
            ]
        )

        node_rules.to_excel(writer, index=False, sheet_name="節點潛力規則")
        link_rules.to_excel(writer, index=False, sheet_name="串聯潛力規則")
        role_rules.to_excel(writer, index=False, sheet_name="功能角色規則")

    return output.getvalue()


# ============================================================
# 八、地圖
# ============================================================

def build_map(df: pd.DataFrame, color_by: str):
    if pdk is None:
        st.warning("尚未安裝 pydeck，無法顯示互動地圖。請在 requirements.txt 加入 pydeck。")
        return

    map_df = df.dropna(subset=["緯度", "經度"]).copy()
    map_df = map_df[(map_df["緯度"].between(21, 27)) & (map_df["經度"].between(118, 123))]

    if map_df.empty:
        st.warning("沒有有效座標可顯示。請確認 TWD97X、TWD97Y 欄位是否正確。")
        return

    if color_by == "功能情境":
        map_df["color"] = map_df["地圖角色代碼"].apply(lambda x: ROLE_COLORS.get(x, ROLE_COLORS["未判定"]))
    elif color_by == "串聯潛力":
        map_df["color"] = map_df["串聯潛力"].apply(lambda x: LINK_COLORS.get(x, [127, 140, 141, 160]))
    else:
        map_df["color"] = map_df["節點潛力"].apply(lambda x: NODE_COLORS.get(x, [127, 140, 141, 160]))

    map_df["radius"] = np.sqrt(map_df["估算可植栽面積(公頃)"].fillna(0.2).clip(lower=0.2, upper=60)) * 120

    view_state = pdk.ViewState(
        longitude=float(map_df["經度"].mean()),
        latitude=float(map_df["緯度"].mean()),
        zoom=7,
        pitch=0,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[經度, 緯度]",
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": """
        <b>{基地名稱}</b><br/>
        縣市：{縣市}<br/>
        鄉鎮市區：{鄉鎮市區}<br/>
        類型：{基地類型(校正)}<br/>
        節點潛力：{節點潛力}（{節點潛力分數}/20）<br/>
        串聯潛力：{串聯潛力}（{串聯潛力分數}/8）<br/>
        可能功能情境：{可能功能情境}<br/>
        適用性：{植生措施適用性}<br/>
        建議：{優先推動建議}
        """,
        "style": {"backgroundColor": "white", "color": "black"},
    }

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip,
        ),
        use_container_width=True,
    )


# ============================================================
# 九、Streamlit 介面
# ============================================================

st.title("🌿 城市綠網空品植生分析工具")
st.caption("新版規則：節點潛力 20 分、串聯潛力 8 分，功能角色以 A1–C3 多情境自動判釋。")

with st.sidebar:
    st.header("必要欄位")
    st.write("上傳檔案需包含以下欄位：")
    st.dataframe(pd.DataFrame({"欄位名稱": REQUIRED_COLUMNS}), hide_index=True, use_container_width=True)

    st.divider()
    st.header("規則摘要")
    st.markdown(
        """
        **節點潛力**：10 項指標，每項 0–2 分，滿分 20 分。  
        **串聯潛力**：4 項指標，每項 0–2 分，滿分 8 分。  
        **功能角色**：依現況是／否欄位與道路距離，自動列出可能符合的 A1–C3 情境。
        """
    )

uploaded_file = st.file_uploader("請上傳你的綠化單元檔案，可使用 Excel 或 CSV", type=["xlsx", "xls", "csv"])

if uploaded_file is None:
    st.info("請上傳 Excel 或 CSV 檔案後開始分析。")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = read_csv_with_fallback(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
except ImportError:
    st.error(
        "目前環境缺少 openpyxl，無法讀取 Excel。"
        "你可以安裝 openpyxl，或先把 Excel 另存成 CSV 後再上傳。"
    )
    st.stop()
except Exception as e:
    st.error(f"讀取檔案失敗：{e}")
    st.stop()

raw_df = normalize_columns(raw_df)

st.subheader("一、欄位檢查")
is_valid, missing = check_required_columns(raw_df)

if not is_valid:
    st.error("欄位檢查未通過，請補齊以下欄位後重新上傳。")
    st.code("\n".join(missing))
    with st.expander("查看目前讀到的欄位"):
        st.write(list(raw_df.columns))
    st.stop()

st.success("欄位檢查通過。")

with st.expander("查看原始資料"):
    st.dataframe(raw_df, use_container_width=True, hide_index=True)

result_df = analyze_green_network(raw_df)

st.subheader("二、分析摘要")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("總筆數", f"{len(result_df):,}")
col2.metric("高節點潛力", f"{(result_df['節點潛力'] == '高潛力').sum():,}")
col3.metric("高串聯潛力", f"{(result_df['串聯潛力'] == '高潛力').sum():,}")
col4.metric("具邊界情境", f"{result_df['可能功能情境'].str.contains('A3|B3|C3', regex=True, na=False).sum():,}")
col5.metric("縣市數", f"{result_df['縣市'].nunique():,}")

chart_col1, chart_col2, chart_col3 = st.columns(3)
with chart_col1:
    st.write("節點潛力分布")
    st.bar_chart(result_df["節點潛力"].value_counts().reindex(["高潛力", "中潛力", "基礎潛力"]).fillna(0))
    st.caption("節點潛力反映基地作為優先投入據點的條件，依空品壓力、敏感受體、人口暴露、高溫風險、管理條件與公共服務性等 10 項指標加總判定。")
with chart_col2:
    st.write("串聯潛力分布")
    st.bar_chart(result_df["串聯潛力"].value_counts().reindex(["高潛力", "中潛力", "基礎潛力"]).fillna(0))
    st.caption("串聯潛力反映基地與周邊綠化單元形成連接的可能性，依最近綠化距離、500 公尺與 1000 公尺內綠化單元數及周邊綠化總面積判定。")
with chart_col3:
    st.write("功能情境出現次數")
    role_counts = {}
    for roles_text in result_df["可能功能情境"]:
        if roles_text == "未判定":
            role_counts["未判定"] = role_counts.get("未判定", 0) + 1
        else:
            for r in str(roles_text).split("、"):
                role_counts[r] = role_counts.get(r, 0) + 1
    st.bar_chart(pd.Series(role_counts).sort_index())
    st.caption("功能情境可同時出現多類，係依基地現況問題自動判釋 A1–C3，作為後續植生功能與管理方式選擇的參考。")

st.subheader("三、分析結果表")
filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    selected_cities = st.multiselect(
        "篩選縣市",
        options=sorted(result_df["縣市"].dropna().unique()),
        default=sorted(result_df["縣市"].dropna().unique()),
    )
with filter_col2:
    selected_node_levels = st.multiselect(
        "篩選節點潛力",
        options=["高潛力", "中潛力", "基礎潛力"],
        default=["高潛力", "中潛力", "基礎潛力"],
    )
with filter_col3:
    selected_link_levels = st.multiselect(
        "篩選串聯潛力",
        options=["高潛力", "中潛力", "基礎潛力"],
        default=["高潛力", "中潛力", "基礎潛力"],
    )

filtered_df = result_df[
    result_df["縣市"].isin(selected_cities)
    & result_df["節點潛力"].isin(selected_node_levels)
    & result_df["串聯潛力"].isin(selected_link_levels)
].copy()

main_columns = [
    "代碼",
    "縣市",
    "鄉鎮市區",
    "基地名稱",
    "基地類型(校正)",
    "估算可植栽面積(公頃)",
    "節點潛力分數",
    "節點潛力",
    "串聯潛力分數",
    "串聯潛力",
    "可能功能情境",
    "功能情境名稱",
    "主要對應功能",
    "植生措施適用性",
    "建議後續管理方式",
    "追蹤重點",
    "優先推動建議",
]

st.dataframe(filtered_df[main_columns], use_container_width=True, hide_index=True)

st.subheader("四、地圖展示")
map_color_by = st.radio("地圖顏色依據", ["節點潛力", "串聯潛力", "功能情境"], horizontal=True)
build_map(filtered_df, color_by=map_color_by)

st.subheader("五、匯出結果")
excel_bytes = make_excel_download(result_df)
if excel_bytes is not None:
    st.download_button(
        label="匯出完整分析結果 Excel",
        data=excel_bytes,
        file_name="城市綠網空品植生分析結果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
else:
    st.warning("目前環境沒有 openpyxl，暫時提供 CSV 匯出。若要匯出 Excel，請在 requirements.txt 加入 openpyxl。")

st.download_button(
    label="匯出完整分析結果 CSV",
    data=make_csv_download(result_df),
    file_name="城市綠網空品植生分析結果.csv",
    mime="text/csv",
)

with st.expander("查看節點潛力評分明細"):
    node_detail_columns = [
        "代碼",
        "基地名稱",
        "節點_空品壓力分數",
        "節點_敏感受體分數",
        "節點_人口暴露分數",
        "節點_高溫風險分數",
        "節點_土地權屬分數",
        "節點_管理權責分數",
        "節點_可植栽空間分數",
        "節點_短期推動分數",
        "節點_開放可及分數",
        "節點_生活節點分數",
        "節點潛力分數",
        "節點潛力",
    ]
    st.dataframe(result_df[node_detail_columns], use_container_width=True, hide_index=True)

with st.expander("查看串聯潛力評分明細"):
    link_detail_columns = [
        "代碼",
        "基地名稱",
        "串聯_最近綠化距離分數",
        "串聯_500公尺綠化單元分數",
        "串聯_1000公尺綠化單元分數",
        "串聯_1000公尺綠化面積分數",
        "串聯潛力分數",
        "串聯潛力",
    ]
    st.dataframe(result_df[link_detail_columns], use_container_width=True, hide_index=True)

with st.expander("查看功能角色判釋規則"):
    st.dataframe(
        pd.DataFrame([{"功能情境": code, **info} for code, info in ROLE_INFO.items()]),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "提醒：本工具目前為初版規則引擎，適合做原型測試、場址排序與初步盤點。"
    "正式決策前仍建議加入專家審查、現地查核、資料校正與維護管理檢核。"
)
