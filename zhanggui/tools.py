"""Agent 工具集 (对应方案 7.3 Agent 工具集 + 7.3.1 工具清单 + 7.3.2 tool calling)。

7 个工具, 每个实现 Tool 协议 (name/description/schema/run):
  - QueryNearingExpiry        : 临期 SKU (days_to_expiry 1..3)
  - QueryHighValueMembers     : 高价值会员 (member_id/rfm_score/last_visit)
  - QueryWeather              : 当日天气
  - QueryHoliday              : 当日节日
  - GenerateCoupon            : 生成券 (先过 guardrail, 不过返回 margin_violation)
  - DraftGroupMessage         : 群发文案 (≤40字模板 + send_window)
  - DraftMomentsPost          : 朋友圈文案 + 海报模板

设计要点 (忠实复刻方案 7.3):
- 工具不直接生成价格 (护栏兜底), 仅做受众筛选/文案润色/预期效果估算;
  generate_coupon 是唯一例外, 内部必须先过 guardrail.guardrail_ok,
  不过则返回 {"error":"margin_violation","min_discount":...} 让 Agent 改用区间下限,
  杜绝幻觉亏钱 (方案 7.3.1 末段 + 7.2.3)。
- 工具只从 ctx 读数据 (ctx['bundle']/['tenant_id']/['playbook']), 不访问外部 IO,
  体现「工具不编造数据」(方案 7.4.1 system prompt 硬约束)。
- schema 为 OpenAI function-calling 的 JSON Schema dict, 供 LLMBrain 暴露给模型。
- TOOLS = {t.name: t for t in [7 个实例]}, 模块级注册表。

严格遵循 CONTRACT (zhanggui/types.py), 包内用相对 import。
零第三方依赖, 仅标准库。
"""

from typing import Any, Dict, List, Optional

from .guardrail import guardrail_ok, min_discount_floor
from .playbook import get_playbook
from .types import SignalBundle


# === Tool 协议 (对应方案 7.3 type Tool interface) ===

class Tool:
    """工具基类 / 协议: name/description/schema/run。

    子类覆写 name/description/schema/run; 基类提供 _safe helpers。
    ctx 是 dict, 至少含:
      - 'bundle'   : SignalBundle
      - 'tenant_id': int
      - 'playbook' : Playbook (dict 或对象, 见 _playbook_templates)
    工具从 ctx 读数据, 不访问外部。
    """

    name: str = ""
    description: str = ""

    def schema(self) -> Dict[str, Any]:
        """OpenAI function-calling 的 parameters JSON Schema。子类覆写。"""
        return {"type": "object", "properties": {}, "required": []}

    def run(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具, 返回 dict 结果。子类覆写。"""
        raise NotImplementedError

    # ---- ctx 安全读取 helpers (缺字段不抛, 降级为默认) ----

    @staticmethod
    def _bundle(ctx: Dict[str, Any]) -> Optional[SignalBundle]:
        """从 ctx 取 bundle, 缺失或类型错 -> None。"""
        bundle = ctx.get("bundle") if isinstance(ctx, dict) else None
        return bundle if isinstance(bundle, SignalBundle) else None

    @staticmethod
    def _tenant_id(ctx: Dict[str, Any]) -> int:
        """从 ctx 取 tenant_id, 缺失 -> 0。"""
        if not isinstance(ctx, dict):
            return 0
        try:
            return int(ctx.get("tenant_id") or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _playbook_templates(ctx: Dict[str, Any], business: str) -> Dict[str, str]:
        """从 ctx['playbook'] 取 copy_templates dict。

        ctx['playbook'] 可能是:
          - dict (PLAYBOOKS[business] 的值, 含 'copy_templates' key) -> 取之;
          - Playbook 对象 (有 .copy_templates 属性) -> 取之;
        任一缺失回退 get_playbook(business) 的 copy_templates, 永不返回 None。
        """
        if not isinstance(ctx, dict):
            return get_playbook(business).get("copy_templates", {})
        pb = ctx.get("playbook")
        templates: Any = None
        if isinstance(pb, dict):
            templates = pb.get("copy_templates")
        elif pb is not None and hasattr(pb, "copy_templates"):
            templates = getattr(pb, "copy_templates", None)
        if not isinstance(templates, dict) or not templates:
            templates = get_playbook(business).get("copy_templates", {})
        return templates

    @staticmethod
    def _fill_template(tpl: str, mapping: Dict[str, Any]) -> str:
        """安全 str.format 填充模板: 缺 key 用空串, 不抛 KeyError。

        模板里的 {foo} 若 mapping 无 foo 则替换为空串 (避免 KeyError 中断)。
        """
        if not isinstance(tpl, str) or not tpl:
            return ""
        try:
            # 用 SafeDict: 缺 key 返回空串而非抛错。
            return tpl.format_map(_SafeDict(mapping))
        except (IndexError, ValueError):
            # 模板含 {0} 之类位置占位或非法 format spec, 退化为原模板。
            return tpl


class _SafeDict(dict):
    """str.format_map 用的安全 dict: 缺 key 返回空串, 不抛 KeyError。"""

    def __missing__(self, key):
        return ""


# === 1. QueryNearingExpiry ===

class QueryNearingExpiry(Tool):
    """查临期 SKU: 过滤 days_to_expiry 1..3 的库存项。

    入参 (方案 7.3.1): {tenant_id, days}。days 缺省取 3, 范围 [1, days]。
    出参: [{product_id,name,cost,price,stock,days_to_expiry}]。
    数据来源: ctx['bundle'].inventory (PG inventory 表临期标记的等价模拟)。
    """

    name = "query_nearing_expiry"
    description = "查询临期商品(days_to_expiry 1-3天),用于临期清仓券选品"

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "integer"},
                "days": {"type": "integer", "description": "临期天数上限,默认3"},
            },
            "required": ["tenant_id"],
        }

    def run(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        bundle = self._bundle(ctx)
        if bundle is None:
            return {"items": []}
        # days 缺省 3, 钳到 [1, 30] (合理上限, 防幻觉大范围)。
        try:
            days = int((args or {}).get("days") or 3)
        except (TypeError, ValueError):
            days = 3
        if days < 1:
            days = 1
        if days > 30:
            days = 30

        out: List[Dict[str, Any]] = []
        for inv in bundle.inventory or []:
            dte = int(getattr(inv, "days_to_expiry", -1))
            # -1 = 非临期; 只取 1..days 的真临期项。
            if dte < 1 or dte > days:
                continue
            out.append({
                "product_id": int(getattr(inv, "product_id", 0)),
                "name": str(getattr(inv, "name", "") or ""),
                "cost": float(getattr(inv, "cost", 0.0) or 0.0),
                "price": float(getattr(inv, "price", 0.0) or 0.0),
                "stock": float(getattr(inv, "stock", 0.0) or 0.0),
                "days_to_expiry": dte,
            })
        return {"items": out}


# === 2. QueryHighValueMembers ===

class QueryHighValueMembers(Tool):
    """查高价值会员: 返回 [{member_id,rfm_score,last_visit}]。

    入参 (方案 7.3.1): {tenant_id, days_inactive_min, limit}。
    出参: [{member_id,name_tag,rfm_score,last_visit}]。
    数据来源: ctx['bundle'].members.high_value (PG member + RFM 视图的等价模拟)。
    本 demo 无真实 RFM, rfm_score 用 member_id 末位做确定性占位 (无 LLM 幻觉)。
    """

    name = "query_high_value_members"
    description = "查询高价值会员列表(含RFM评分),用于沉睡唤醒/高毛利push受众筛选"

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "integer"},
                "days_inactive_min": {"type": "integer", "description": "沉睡天数下限,默认30"},
                "limit": {"type": "integer", "description": "返回上限,默认50"},
            },
            "required": ["tenant_id"],
        }

    def run(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        bundle = self._bundle(ctx)
        if bundle is None:
            return {"items": [], "total": 0}
        members = getattr(getattr(bundle, "members", None), "high_value", None) or []
        # limit 钳到 [1, 200] (防幻觉超大范围)。
        try:
            limit = int((args or {}).get("limit") or 50)
        except (TypeError, ValueError):
            limit = 50
        if limit < 1:
            limit = 1
        if limit > 200:
            limit = 200

        out: List[Dict[str, Any]] = []
        for mid in members[:limit]:
            try:
                mid_i = int(mid)
            except (TypeError, ValueError):
                continue
            # rfm_score 确定性占位: 末位*0.1+4.0, 区间约 [4.0,4.9], 不编造真实 RFM。
            rfm = round(4.0 + (mid_i % 10) * 0.1, 2)
            out.append({
                "member_id": mid_i,
                "name_tag": "高价值会员#{m}".format(m=mid_i),
                "rfm_score": rfm,
                "last_visit": "2026-06-{d:02d}".format(d=1 + (mid_i % 28)),
            })
        return {"items": out, "total": len(out)}


# === 3. QueryWeather ===

class QueryWeather(Tool):
    """查当日天气: 返回 {condition,temp_c,forecast}。

    入参 (方案 7.3.1): {city, date}。本 demo 从 ctx['bundle'].weather 直取。
    出参: {condition,temp_c,forecast}。
    """

    name = "query_weather"
    description = "查询商户当日天气(条件/温度/预报),用于雨天配送/天气营销判断"

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": [],
        }

    def run(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        bundle = self._bundle(ctx)
        if bundle is None:
            return {"condition": "normal", "temp_c": 20.0, "forecast": ""}
        w = getattr(bundle, "weather", None)
        if w is None:
            return {"condition": "normal", "temp_c": 20.0, "forecast": ""}
        return {
            "condition": str(getattr(w, "condition", "normal") or "normal"),
            "temp_c": float(getattr(w, "temp_c", 20.0) or 0.0),
            "forecast": str(getattr(w, "forecast", "") or ""),
        }


# === 4. QueryHoliday ===

class QueryHoliday(Tool):
    """查当日节日: 返回 {name,days_to,is_festival}。

    入参 (方案 7.3.1): {date}。本 demo 从 ctx['bundle'].holiday 直取。
    出参: {name,days_to,is_festival}。
    """

    name = "query_holiday"
    description = "查询当日/临近节日(名称/距今天数/是否大促节日),用于节日满减判断"

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
            },
            "required": [],
        }

    def run(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        bundle = self._bundle(ctx)
        if bundle is None:
            return {"name": "无", "days_to": -1, "is_festival": False}
        h = getattr(bundle, "holiday", None)
        if h is None:
            return {"name": "无", "days_to": -1, "is_festival": False}
        return {
            "name": str(getattr(h, "name", "无") or "无"),
            "days_to": int(getattr(h, "days_to", -1)),
            "is_festival": bool(getattr(h, "is_festival", False)),
        }


# === 5. GenerateCoupon ===

class GenerateCoupon(Tool):
    """生成券: 先过 guardrail, 不过返回 margin_violation。

    入参 (方案 7.3.1): {tenant_id,product,discount,limit_qty,expire_hours}。
    出参 (通过): {coupon_tpl_id,face_value,valid_until}。
    出参 (护栏不过): {"error":"margin_violation","min_discount":...}。
    数据来源: 从 ctx['bundle'].inventory 查 product 的 cost/price, 再调
      guardrail.guardrail_ok(cost, price*discount, margin_floor);
      margin_floor 从 ctx['margin_floor'] 取 (Agent 段注入), 缺省 0.0。
    face_value = price*(1-discount) (券面值 = 让利额)。
    """

    name = "generate_coupon"
    description = "生成优惠券(先过毛利护栏,不过返回margin_violation让Agent改用区间下限)"

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "integer"},
                "product": {"type": "string", "description": "商品名"},
                "discount": {"type": "number", "description": "折扣(0-1,如0.6=6折)"},
                "limit_qty": {"type": "integer", "description": "限份数"},
                "expire_hours": {"type": "integer", "description": "有效小时数"},
            },
            "required": ["product", "discount"],
        }

    def run(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            args = {}
        bundle = self._bundle(ctx)
        if bundle is None:
            return {"error": "no_bundle"}
        product = str(args.get("product") or "")
        # discount 安全解析, 钳到 [0, 1]。
        try:
            discount = float(args.get("discount"))
        except (TypeError, ValueError):
            return {"error": "invalid_discount"}
        if discount < 0.0:
            discount = 0.0
        if discount > 1.0:
            discount = 1.0

        # 从 inventory 查 product 的 cost/price (不编造, 方案 7.4.1 硬约束)。
        inv = None
        for it in bundle.inventory or []:
            if str(getattr(it, "name", "") or "") == product:
                inv = it
                break
        if inv is None:
            return {"error": "product_not_found", "product": product}

        cost = float(getattr(inv, "cost", 0.0) or 0.0)
        price = float(getattr(inv, "price", 0.0) or 0.0)
        if cost <= 0 or price <= 0:
            # 无成本/售价数据不给出价动作 (防幻觉亏钱, 方案 7.2.3)。
            return {"error": "no_cost_data"}

        # margin_floor 从 ctx 取 (Agent 段注入 rule 的 margin_floor), 缺省 0.0。
        try:
            margin_floor = float(ctx.get("margin_floor") if isinstance(ctx, dict) else 0.0)
        except (TypeError, ValueError):
            margin_floor = 0.0

        final_price = price * discount
        # 护栏硬约束: final_price >= cost*(1+margin_floor)。
        if not guardrail_ok(cost, final_price, margin_floor):
            min_disc = min_discount_floor(cost, price, margin_floor)
            # 钳到 [0,1] 展示 (inf 表示无法给出有意义折扣)。
            min_disc_disp = min_disc if 0.0 <= min_disc <= 1.0 else 1.0
            return {
                "error": "margin_violation",
                "min_discount": round(min_disc_disp, 4),
                "cost": round(cost, 2),
                "price": round(price, 2),
                "margin_floor": margin_floor,
            }

        # 通过: 产出券模板占位 (不写真实 DB, demo 等价模拟)。
        face_value = round(price * (1.0 - discount), 2)
        try:
            limit_qty = int(args.get("limit_qty") or 0)
        except (TypeError, ValueError):
            limit_qty = 0
        try:
            expire_hours = int(args.get("expire_hours") or 0)
        except (TypeError, ValueError):
            expire_hours = 0
        coupon_tpl_id = "cpn_{p}_{d}".format(
            p=product[:6].replace(" ", "_") or "x", d=int(discount * 100)
        )
        valid_until = "{h}h".format(h=expire_hours) if expire_hours > 0 else "今日有效"
        return {
            "coupon_tpl_id": coupon_tpl_id,
            "face_value": face_value,
            "valid_until": valid_until,
            "discount": discount,
            "limit_qty": limit_qty,
            "final_price": round(final_price, 2),
        }


# === 6. DraftGroupMessage ===

class DraftGroupMessage(Tool):
    """群发文案: 从 playbook 模板填充, 返回 {copy, send_window}。

    入参 (方案 7.3.1): {tenant_id,group_id,action_type,product,discount}。
    出参: {copy(≤40字), send_window}。copy 从 ctx['playbook'].copy_templates
      按 action_type 取模板, 用入参 mapping 填充, 截断 ≤40 字。
    send_window 默认 "19:00" (方案 7.4.2 示例「今晚7点群秒杀」)。
    """

    name = "draft_group_message"
    description = "起草社群群发文案(≤40字模板填充),含发送时段send_window"

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "integer"},
                "group_id": {"type": "string"},
                "action_type": {"type": "string"},
                "product": {"type": "string"},
                "discount": {"type": "number"},
                "price": {"type": "number"},
                "qty": {"type": "integer"},
                "spend": {"type": "number"},
                "reduce": {"type": "number"},
                "hours": {"type": "integer"},
            },
            "required": ["action_type"],
        }

    # 群发默认时段 (方案 7.4.2「今晚7点群秒杀」)。
    _DEFAULT_WINDOW = "19:00"

    def run(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            args = {}
        bundle = self._bundle(ctx)
        business = str(getattr(bundle, "business", "") or "") if bundle else ""
        action_type = str(args.get("action_type") or "")

        templates = self._playbook_templates(ctx, business)
        tpl = templates.get(action_type)
        if not tpl:
            # 回退通用模板 (≤40 字)。
            tpl = "今日推荐{product} 速来抢购"

        mapping = self._build_mapping(args, ctx)
        copy = self._fill_template(tpl, mapping)
        copy = self._truncate(copy, 40)
        return {"copy": copy, "send_window": self._DEFAULT_WINDOW}

    @staticmethod
    def _build_mapping(args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """把工具入参 + ctx 折成模板占位符 mapping。

        覆盖 playbook 模板常用占位: product/price/qty/discount/spend/reduce/
        hours/member_tag/weather_emoji/audience_size/combo/save/reduce2。
        缺省值用空串或 0, 不抛错。
        """
        def _g(k, default=""):
            v = args.get(k, default)
            return v if v is not None else default

        bundle = Tool._bundle(ctx)
        weather_cond = ""
        if bundle is not None and getattr(bundle, "weather", None) is not None:
            weather_cond = str(getattr(bundle.weather, "condition", "") or "")

        _EMOJI = {"rain": "🌧", "hot": "☀️", "cold": "❄️", "snow": "⛄", "normal": ""}

        # 价格展示: discount 折算后的 price_after 优先; 否则取入参 price。
        price_raw = _g("price", "")
        try:
            disc = float(_g("discount", 0) or 0)
        except (TypeError, ValueError):
            disc = 0.0
        if price_raw not in ("", 0):
            try:
                pa = float(price_raw) * (disc if 0 < disc <= 1 else 1.0)
                price_disp = "{0:.1f}".format(pa)
            except (TypeError, ValueError):
                price_disp = str(price_raw)
        else:
            price_disp = str(price_raw)

        # qty / spend / reduce / hours 展示为整型或保留 1 位。
        def _num(k, default=""):
            v = _g(k, default)
            try:
                f = float(v)
                if f == int(f):
                    return str(int(f))
                return "{0:.1f}".format(f)
            except (TypeError, ValueError):
                return str(v) if v != "" else ""

        return {
            "product": str(_g("product", "")),
            "price": price_disp,
            "qty": _num("qty"),
            "discount": _num("discount"),
            "spend": _num("spend"),
            "reduce": _num("reduce"),
            "reduce2": _num("reduce"),
            "hours": _num("hours"),
            "member_tag": str(_g("member_tag", "会员") or "会员"),
            "audience_size": _num("audience_size"),
            "combo": str(_g("combo", "畅销款") or "畅销款"),
            "save": _num("reduce", ""),
            "weather_emoji": _EMOJI.get(weather_cond, ""),
        }

    @staticmethod
    def _truncate(s: str, max_len: int = 40) -> str:
        """按字符数截断 ≤ max_len (中文每字 1)。"""
        if not isinstance(s, str):
            return ""
        if len(s) <= max_len:
            return s
        return s[:max_len]


# === 7. DraftMomentsPost ===

class DraftMomentsPost(Tool):
    """朋友圈文案: 从 playbook 模板填充, 返回 {copy, image_tpl_id}。

    入参 (方案 7.3.1): {tenant_id,action_type,product}。
    出参: {copy, image_tpl_id}。image_tpl_id 固定 "tpl_poster_1"
      (demo 无真实 COS 海报模板, 等价占位)。
    """

    name = "draft_moments_post"
    description = "起草朋友圈文案+海报模板(≤40字模板填充),image_tpl_id固定tpl_poster_1"

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "integer"},
                "action_type": {"type": "string"},
                "product": {"type": "string"},
                "discount": {"type": "number"},
                "price": {"type": "number"},
            },
            "required": ["action_type"],
        }

    _IMAGE_TPL = "tpl_poster_1"

    def run(self, args: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            args = {}
        bundle = self._bundle(ctx)
        business = str(getattr(bundle, "business", "") or "") if bundle else ""
        action_type = str(args.get("action_type") or "")

        templates = self._playbook_templates(ctx, business)
        tpl = templates.get(action_type)
        if not tpl:
            tpl = "今日推荐{product} 朋友圈专享速抢"

        # 复用 DraftGroupMessage 的 mapping 构造 (避免重复实现)。
        mapping = DraftGroupMessage._build_mapping(args, ctx)
        copy = self._fill_template(tpl, mapping)
        copy = DraftGroupMessage._truncate(copy, 40)
        return {"copy": copy, "image_tpl_id": self._IMAGE_TPL}


# === 注册表 (对应方案 7.3 var registry) ===

TOOLS: Dict[str, Tool] = {t.name: t for t in [
    QueryNearingExpiry(),
    QueryHighValueMembers(),
    QueryWeather(),
    QueryHoliday(),
    GenerateCoupon(),
    DraftGroupMessage(),
    DraftMomentsPost(),
]}


def schemas() -> List[Dict[str, Any]]:
    """把 TOOLS 暴露成 OpenAI function-calling 的 tools 数组 (方案 7.3 schemas())。"""
    out: List[Dict[str, Any]] = []
    for t in TOOLS.values():
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema(),
            },
        })
    return out


__all__ = [
    "Tool",
    "QueryNearingExpiry",
    "QueryHighValueMembers",
    "QueryWeather",
    "QueryHoliday",
    "GenerateCoupon",
    "DraftGroupMessage",
    "DraftMomentsPost",
    "TOOLS",
    "schemas",
]
