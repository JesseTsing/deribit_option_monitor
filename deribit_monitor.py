import asyncio
import json
import panel as pn
import pandas as pd
import numpy as np
import websockets
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.stats import norm

# 加载 Panel 扩展
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")

# ==========================================
# 1. Black-Scholes 期权定价与全阶希腊字母分布计算
# ==========================================
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """BS 定价模型"""
    T = max(T, 0.00001)
    sigma = max(sigma, 0.0001)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_greeks_profile(legs, s_range, r=0.02, days_forward=0):
    """
    计算组合在标的价格序列 s_range 上的 1阶、2阶、3阶全套希腊字母分布
    """
    num_points = len(s_range)
    greeks = {
        'Delta': np.zeros(num_points),
        'Vega': np.zeros(num_points),
        'Gamma': np.zeros(num_points),
        'Theta': np.zeros(num_points),
        'Vanna': np.zeros(num_points),
        'Volga': np.zeros(num_points),
        'Charm': np.zeros(num_points),
        'Speed': np.zeros(num_points),
        'Zomma': np.zeros(num_points),
        'Color': np.zeros(num_points),
    }

    if not legs:
        return greeks

    for leg in legs:
        K = float(leg['strike'])
        qty = float(leg['qty'])
        opt_type = leg['type'].lower()
        iv = max(0.0001, leg['iv'])
        
        dte = max(leg['dte_base'] - days_forward, 0.001)
        T = dte / 365.0
        sqrt_T = np.sqrt(T)

        d1 = (np.log(s_range / K) + (r + 0.5 * iv ** 2) * T) / (iv * sqrt_T)
        d2 = d1 - iv * sqrt_T
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d2 = norm.cdf(-d2)

        # 1st Order
        if opt_type == 'call':
            delta = cdf_d1
            theta_annual = -(s_range * pdf_d1 * iv) / (2 * sqrt_T) - r * K * np.exp(-r * T) * cdf_d2
            charm_annual = -pdf_d1 * (r / (iv * sqrt_T) - d2 / (2 * T)) - r * np.exp(-r * T) * cdf_d2
        else:
            delta = cdf_d1 - 1.0
            theta_annual = -(s_range * pdf_d1 * iv) / (2 * sqrt_T) + r * K * np.exp(-r * T) * cdf_neg_d2
            charm_annual = -pdf_d1 * (r / (iv * sqrt_T) - d2 / (2 * T)) + r * np.exp(-r * T) * cdf_neg_d2

        vega_1pct = (s_range * pdf_d1 * sqrt_T) / 100.0

        # 2nd Order
        gamma = pdf_d1 / (s_range * iv * sqrt_T)
        theta_daily = theta_annual / 365.0
        charm_daily = charm_annual / 365.0
        vanna_1pct = (-pdf_d1 * d2 / iv) / 100.0
        volga_1pct = (vega_1pct * 100.0 * d1 * d2 / iv) / 100.0

        # 3rd Order
        speed = - (gamma / s_range) * (d1 / (iv * sqrt_T) + 1.0)
        zomma_1pct = (gamma * (d1 * d2 - 1.0) / iv) / 100.0
        color_daily = (-gamma * (r * d1 / (iv * sqrt_T) + (1.0 - d1 * d2) / (2.0 * T))) / 365.0

        # 组合累加
        greeks['Delta'] += delta * qty
        greeks['Vega'] += vega_1pct * qty
        greeks['Gamma'] += gamma * qty
        greeks['Theta'] += theta_daily * qty
        greeks['Vanna'] += vanna_1pct * qty
        greeks['Volga'] += volga_1pct * qty
        greeks['Charm'] += charm_daily * qty
        greeks['Speed'] += speed * qty
        greeks['Zomma'] += zomma_1pct * qty
        greeks['Color'] += color_daily * qty

    return greeks

# ==========================================
# 2. Deribit WebSocket 客户端
# ==========================================
class DeribitWSClient:
    def __init__(self, ws_url="wss://www.deribit.com/ws/api/v2"):
        self.ws_url = ws_url
        self.ws = None
        self.req_id = 0
        self.pending_requests = {}
        
        self.instruments = {}
        self.market_data = {}
        self.subscribed_channels = set()
        self.current_underlying_price = 0.0
        self.is_connected = False

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20)
        self.is_connected = True
        asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                if "id" in data and data["id"] in self.pending_requests:
                    future = self.pending_requests.pop(data["id"])
                    if not future.done():
                        future.set_result(data.get("result"))
                elif data.get("method") == "subscription":
                    params = data.get("params", {})
                    channel = params.get("channel", "")
                    data_payload = params.get("data", {})
                    if channel.startswith("ticker."):
                        inst_name = data_payload.get("instrument_name")
                        if inst_name:
                            self.market_data[inst_name] = data_payload
                            u_price = data_payload.get("underlying_price") or 0.0
                            if u_price > 0:
                                self.current_underlying_price = u_price
        except Exception as e:
            print(f"WebSocket 监听异常: {e}")
            self.is_connected = False

    async def send_rpc(self, method, params=None):
        if not self.ws or not self.is_connected:
            await self.connect()
        self.req_id += 1
        req_id = self.req_id
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_requests[req_id] = future
        await self.ws.send(json.dumps(payload))
        return await future

    async def get_active_option_instruments(self, currency="BTC"):
        res = await self.send_rpc("public/get_instruments", {"currency": currency, "kind": "option", "expired": False})
        if not res:
            return [], []
        self.instruments = {item["instrument_name"]: item for item in res}
        expirations = set()
        for item in res:
            parts = item["instrument_name"].split("-")
            if len(parts) >= 4:
                expirations.add(parts[1])
        def parse_exp(exp_str):
            try: return datetime.strptime(exp_str, "%d%b%y")
            except: return datetime.max
        return res, sorted(list(expirations), key=parse_exp)

    async def update_subscriptions(self, target_instruments):
        new_channels = set([f"ticker.{inst}.100ms" for inst in target_instruments])
        to_unsubscribe = list(self.subscribed_channels - new_channels)
        to_subscribe = list(new_channels - self.subscribed_channels)
        if to_unsubscribe:
            await self.send_rpc("public/unsubscribe", {"channels": to_unsubscribe})
            for ch in to_unsubscribe: self.subscribed_channels.remove(ch)
        if to_subscribe:
            await self.send_rpc("public/subscribe", {"channels": to_subscribe})
            for ch in to_subscribe: self.subscribed_channels.add(ch)

# ==========================================
# 3. UI 控件配置
# ==========================================
ws_client = DeribitWSClient()
user_position_store = {}

currency_select = pn.widgets.Select(name="基础币种 (Currency)", options=["BTC", "ETH", "SOL"], value="BTC")
expiry_select = pn.widgets.Select(name="到期日 (Expiry)", options=[])
spot_indicator = pn.indicators.Number(name="标的现货价格 (Spot)", value=0.0, format="${value:,.2f}")
status_text = pn.pane.Markdown("**🟡 正在初始化 WebSocket...**")

iv_shift_slider = pn.widgets.FloatSlider(name="IV 波动率调整 (%)", start=-50.0, end=50.0, step=1.0, value=0.0)
days_forward_slider = pn.widgets.IntSlider(name="推移天数 (Days Forward)", start=0, end=60, step=1, value=0)

price_range_slider = pn.widgets.RangeSlider(
    name="标的价格显示区间 ($) [Min: 0]", 
    start=0.0, 
    end=150000.0, 
    value=(0.0, 100000.0), 
    step=250.0
)

# 表格列编辑配置
disabled_cols = ["Strike", "Bid ($)", "Ask ($)", "Bid (Coin)", "Ask (Coin)", "IV", "Delta", "Gamma", "Theta", "Vega", "Volume 24h"]
editors = {col: None for col in disabled_cols}
editors['model'] = {'type': 'number', 'step': 1}
editors['pos'] = {'type': 'number', 'step': 1}

call_table = pn.widgets.Tabulator(pd.DataFrame(), theme='midnight', height=300, show_index=False, editors=editors, disabled=False, configuration={'layout': 'fitDataFill'})
put_table = pn.widgets.Tabulator(pd.DataFrame(), theme='midnight', height=300, show_index=False, editors=editors, disabled=False, configuration={'layout': 'fitDataFill'})

# ==========================================
# 4. 数据构建与刷新逻辑
# ==========================================
def sync_user_positions_from_tables():
    if not call_table.value.empty and 'model' in call_table.value.columns:
        for _, row in call_table.value.iterrows():
            user_position_store[(row['Strike'], 'Call')] = {'model': int(row.get('model', 0) or 0), 'pos': int(row.get('pos', 0) or 0)}
    if not put_table.value.empty and 'model' in put_table.value.columns:
        for _, row in put_table.value.iterrows():
            user_position_store[(row['Strike'], 'Put')] = {'model': int(row.get('model', 0) or 0), 'pos': int(row.get('pos', 0) or 0)}

def build_display_dataframes():
    if not expiry_select.value or not ws_client.instruments:
        return pd.DataFrame(), pd.DataFrame()

    sync_user_positions_from_tables()
    target_exp = expiry_select.value
    rows = []

    for inst_name, info in ws_client.instruments.items():
        parts = inst_name.split("-")
        if len(parts) < 4 or parts[1] != target_exp: continue

        strike = float(parts[2])
        opt_type = "Call" if parts[3] == "C" else "Put"
        data = ws_client.market_data.get(inst_name, {})
        u_price = data.get("underlying_price") or ws_client.current_underlying_price or 0.0
        
        bid_coin, ask_coin = data.get("best_bid_price") or 0.0, data.get("best_ask_price") or 0.0
        greeks = data.get("greeks") or {}
        iv = data.get("mark_iv") or 0.0
        volume = (data.get("stats") or {}).get("volume") or 0.0
        pos_info = user_position_store.get((strike, opt_type), {'model': 0, 'pos': 0})

        rows.append({
            "model": int(pos_info['model']),
            "pos": int(pos_info['pos']),
            "Type": opt_type,
            "Strike": strike,
            "Bid ($)": round(bid_coin * u_price, 2) if (bid_coin and u_price) else 0.0,
            "Ask ($)": round(ask_coin * u_price, 2) if (ask_coin and u_price) else 0.0,
            "Bid (Coin)": round(bid_coin, 4) if bid_coin else 0.0,
            "Ask (Coin)": round(ask_coin, 4) if ask_coin else 0.0,
            "Delta": round(greeks.get("delta"), 3) if greeks.get("delta") is not None else None,
            "Gamma": round(greeks.get("gamma"), 5) if greeks.get("gamma") is not None else None,
            "Theta": round(greeks.get("theta"), 2) if greeks.get("theta") is not None else None,
            "Vega": round(greeks.get("vega"), 2) if greeks.get("vega") is not None else None,
            "IV": f"{iv:.1f}%" if iv else "N/A",
            "Volume 24h": round(volume, 2)
        })

    df = pd.DataFrame(rows)
    if df.empty: return pd.DataFrame(), pd.DataFrame()

    call_df = df[df["Type"] == "Call"].drop(columns=["Type"]).sort_values(by="Strike").reset_index(drop=True)
    put_df = df[df["Type"] == "Put"].drop(columns=["Type"]).sort_values(by="Strike").reset_index(drop=True)
    return call_df, put_df

def extract_current_legs(call_df, put_df):
    dte_base = 30.0
    if expiry_select.value:
        try:
            exp_dt = datetime.strptime(expiry_select.value, "%d%b%y")
            dte_base = max((exp_dt - datetime.utcnow()).total_seconds() / 86400.0, 0.05)
        except: dte_base = 30.0

    legs = []
    for df, opt_type in [(call_df, 'call'), (put_df, 'put')]:
        if df is None or df.empty: continue
        for _, row in df.iterrows():
            m, p = int(row.get('model', 0) or 0), int(row.get('pos', 0) or 0)
            total_qty = m + p
            if total_qty == 0: continue

            strike = float(row['Strike'])
            iv_str = str(row.get('IV', '0')).replace('%', '')
            try: base_iv = float(iv_str) / 100.0 if iv_str != 'N/A' else 0.5
            except: base_iv = 0.5

            legs.append({
                'strike': strike,
                'qty': total_qty,
                'type': opt_type,
                'iv': base_iv,
                'dte_base': dte_base
            })
    return legs, dte_base

# ==========================================
# 5. 图表 1: Risk Graph 收益曲线图 (图例移顶部，X轴硬对齐)
# ==========================================
@pn.depends(call_table.param.value, put_table.param.value, iv_shift_slider.param.value, days_forward_slider.param.value, price_range_slider.param.value)
def plot_risk_graph(call_df, put_df, iv_shift, days_forward, price_range):
    spot = ws_client.current_underlying_price or 1000.0
    fig = go.Figure()

    legs, dte_base = extract_current_legs(call_df, put_df)

    if not legs:
        fig.add_annotation(
            text="请在左侧 Call / Put 表格的 model 或 pos 列输入仓位",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title="期权仓位 Payoff Risk Graph", template="plotly_dark", height=380, margin=dict(l=80, r=40, t=60, b=40))
        return fig

    x_min, x_max = price_range[0], max(price_range[1], price_range[0] + 10.0)
    s_range = np.linspace(max(0.001, x_min), x_max, 250)

    r = 0.02
    T0 = max(dte_base - days_forward, 0.00001) / 365.0
    T7 = max(dte_base - days_forward - 7, 0.00001) / 365.0

    total_pnl_t0 = np.zeros_like(s_range)
    total_pnl_t7 = np.zeros_like(s_range)
    total_pnl_exp = np.zeros_like(s_range)

    for leg in legs:
        strike, qty, opt_type = leg['strike'], leg['qty'], leg['type']
        adj_iv = max(0.01, leg['iv'] + (iv_shift / 100.0))
        cost_price = black_scholes(spot, strike, dte_base / 365.0, r, leg['iv'], opt_type)

        val_t0 = black_scholes(s_range, strike, T0, r, adj_iv, opt_type)
        total_pnl_t0 += (val_t0 - cost_price) * qty

        val_t7 = black_scholes(s_range, strike, T7, r, adj_iv, opt_type)
        total_pnl_t7 += (val_t7 - cost_price) * qty

        val_exp = np.maximum(s_range - strike, 0) if opt_type == 'call' else np.maximum(strike - s_range, 0)
        total_pnl_exp += (val_exp - cost_price) * qty

    fig.add_trace(go.Scatter(x=s_range, y=total_pnl_t0, mode='lines', name='T+0 实时盈亏', line=dict(color='#ab47bc', width=2.5)))
    fig.add_trace(go.Scatter(x=s_range, y=total_pnl_t7, mode='lines', name='T+7 预计盈亏', line=dict(color='#ffa726', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=s_range, y=total_pnl_exp, mode='lines', name='Expiry 到期盈亏', line=dict(color='#29b6f6', width=2, dash='dash')))

    if x_min <= spot <= x_max:
        fig.add_vline(x=spot, line_dash="dash", line_color="#ef5350", annotation_text=f"Spot: ${spot:,.2f}")
    fig.add_hline(y=0, line_color="#78909c", line_width=1)

    fig.update_layout(
        title=f"期权仓位 Payoff Risk Graph (范围: ${x_min:,.0f} ~ ${x_max:,.0f})",
        xaxis_title="标的价格 ($)",
        yaxis_title="盈亏金额 ($)",
        xaxis=dict(range=[x_min, x_max]),
        template="plotly_dark",
        height=380,
        hovermode="x unified",
        # 图例顶部横向排列，不挤压右侧坐标轴宽度
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01
        ),
        # 与下方的全阶希腊字母柱状图严格统一左右边距 (l=80, r=40)
        margin=dict(l=80, r=40, t=60, b=40)
    )
    return fig

# ==========================================
# 6. 图表 2: 全阶希腊字母 (1阶/2阶/3阶) 10行柱状图 (硬对齐 X 轴)
# ==========================================
@pn.depends(call_table.param.value, put_table.param.value, iv_shift_slider.param.value, days_forward_slider.param.value, price_range_slider.param.value)
def plot_greeks_chart(call_df, put_df, iv_shift, days_forward, price_range):
    spot = ws_client.current_underlying_price or 1000.0
    legs, _ = extract_current_legs(call_df, put_df)

    x_min, x_max = price_range[0], max(price_range[1], price_range[0] + 10.0)
    s_range = np.linspace(max(0.001, x_min), x_max, 80)

    for leg in legs:
        leg['iv'] = max(0.01, leg['iv'] + (iv_shift / 100.0))

    greeks_dict = calculate_greeks_profile(legs, s_range, days_forward=days_forward)

    greeks_list = [
        ("1阶-Delta (Δ)", greeks_dict['Delta']),
        ("1阶-Vega (V 1%)", greeks_dict['Vega']),
        ("2阶-Gamma (Γ)", greeks_dict['Gamma']),
        ("2阶-Theta (Θ 每日)", greeks_dict['Theta']),
        ("2阶-Vanna (dΔ/dσ)", greeks_dict['Vanna']),
        ("2阶-Volga (dV/dσ)", greeks_dict['Volga']),
        ("2阶-Charm (dΔ/dt)", greeks_dict['Charm']),
        ("3阶-Speed (dΓ/dS)", greeks_dict['Speed']),
        ("3阶-Zomma (dΓ/dσ)", greeks_dict['Zomma']),
        ("3阶-Color (dΓ/dt)", greeks_dict['Color']),
    ]

    num_rows = len(greeks_list)
    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=[title for title, _ in greeks_list]
    )

    for idx, (title, vals) in enumerate(greeks_list, start=1):
        colors = ['#26a69a' if v >= 0 else '#ef5350' for v in vals]
        
        fig.add_trace(
            go.Bar(
                x=s_range,
                y=vals,
                name=title,
                marker_color=colors,
                showlegend=False
            ),
            row=idx, col=1
        )
        
        fig.add_hline(y=0, line_color="#78909c", line_width=1, row=idx, col=1)
        if x_min <= spot <= x_max:
            fig.add_vline(x=spot, line_dash="dash", line_color="#ef5350", row=idx, col=1)

    fig.update_layout(
        title="全阶希腊字母沿标的价格分布图 (Greeks Exposure vs. Spot Price)",
        template="plotly_dark",
        height=1100,
        hovermode="x unified",
        # 与上方的 Risk Graph 严格统一左右边距 (l=80, r=40)
        margin=dict(l=80, r=40, t=50, b=40)
    )
    
    # 强制将所有子图的 X 轴范围同步为 [x_min, x_max]，并在最底层标注 X 轴标题
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_xaxes(title_text="标的价格 ($)", row=num_rows, col=1)
    return fig

# ==========================================
# 7. 定时刷新与事件响应
# ==========================================
async def refresh_instruments(currency):
    status_text.object = f"**🔄 正在拉取 {currency} 期权链元数据...**"
    insts, expiries = await ws_client.get_active_option_instruments(currency)
    expiry_select.options = expiries
    if expiries:
        expiry_select.value = expiries[0]
        await update_ticker_subscriptions()
    status_text.object = f"**🟢 WebSocket 订阅中** | 频道数: {len(ws_client.subscribed_channels)}"

async def update_ticker_subscriptions():
    if not expiry_select.value: return
    target_exp = expiry_select.value
    matched_insts = [inst for inst, info in ws_client.instruments.items() if len(inst.split("-")) >= 4 and inst.split("-")[1] == target_exp]
    await ws_client.update_subscriptions(matched_insts)
    status_text.object = f"**🟢 实时监控中** | 标的: {currency_select.value} | 到期日: {target_exp} | 订阅合约数: {len(matched_insts)}"

def on_currency_change(event): asyncio.create_task(refresh_instruments(event.new))
def on_expiry_change(event): asyncio.create_task(update_ticker_subscriptions())

currency_select.param.watch(on_currency_change, 'value')
expiry_select.param.watch(on_expiry_change, 'value')

def ui_periodic_timer():
    if ws_client.current_underlying_price > 0:
        spot = ws_client.current_underlying_price
        spot_indicator.value = spot
        
        current_max = max(10000.0, spot * 2.0)
        if price_range_slider.end != current_max:
            price_range_slider.end = round(current_max, -2)
            if price_range_slider.value[1] == 100000.0 or price_range_slider.value[1] > current_max:
                price_range_slider.value = (0.0, round(spot * 1.5, -2))

    call_df, put_df = build_display_dataframes()
    call_table.value = call_df
    put_table.value = put_df

pn.state.add_periodic_callback(ui_periodic_timer, period=10000)

async def init_app():
    await ws_client.connect()
    await refresh_instruments("BTC")

pn.state.onload(lambda: asyncio.ensure_future(init_app()))

# ==========================================
# 8. 界面布局组合
# ==========================================
controls_card = pn.Card(
    currency_select, expiry_select, spot_indicator, status_text,
    title="① 市场选择与状态", 
)

risk_controls_card = pn.Card(
    price_range_slider,
    pn.Row(iv_shift_slider, days_forward_slider),
    title="② Risk Graph 场景与标的区间调控 (Scenario Controls)",
)

left_panel = pn.Column(
    controls_card,
    pn.Card(call_table, title="看涨期权实时盘口 (Calls)", 
    ),
    pn.Card(put_table, title="看跌期权实时盘口 (Puts)", ),
    sizing_mode="stretch_width"
)

right_panel = pn.Column(
    risk_controls_card,
    plot_risk_graph,
    pn.Spacer(height=10),
    plot_greeks_chart,
    sizing_mode="stretch_width"
)

template = pn.template.FastListTemplate(
    title="Deribit 实时期权监控与全阶希腊字母 Risk Graph 终端",
    theme="dark",
    main=[
        pn.Row(
            left_panel,
            right_panel,
            sizing_mode="stretch_both"
        )
    ]
)

template.servable()
