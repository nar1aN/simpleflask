from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
import base64
from typing import Iterable, Optional

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

from models import Candle


class InvestError(RuntimeError):
    pass


# --- Импорты SDK (поддерживаем и новый t-tech-investments, и старый tinkoff-investments) ---
try:
    # Новый SDK (как в документации T-Bank)
    from t_tech.invest import Client, CandleInterval  # type: ignore
    from t_tech.invest.utils import now  # type: ignore

    _SDK_NAME = "t-tech-investments"
except Exception:  # pragma: no cover
    try:
        # Старый SDK (PyPI)
        from tinkoff.invest import Client, CandleInterval  # type: ignore
        from tinkoff.invest.utils import now  # type: ignore

        _SDK_NAME = "tinkoff-investments"
    except Exception as e:  # pragma: no cover
        Client = None  # type: ignore
        CandleInterval = None  # type: ignore
        now = None  # type: ignore
        _SDK_NAME = "not-installed"


@dataclass(frozen=True, slots=True)
class CandleRequest:
    instrument_id: str
    days_back: int = 10
    interval: str = "4h"  # '1m', '5m', '15m', '1h', '4h', '1d'


def _quotation_to_float(q) -> float:
    """В SDK цены обычно приходят как Quotation(units, nano)."""
    units = getattr(q, "units", 0)
    nano = getattr(q, "nano", 0)
    try:
        return float(units) + float(nano) / 1_000_000_000.0
    except Exception:
        return float(units)


def _interval_from_str(interval: str):
    if CandleInterval is None:
        raise InvestError("SDK не установлен")

    m = {
        "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
        "4h": CandleInterval.CANDLE_INTERVAL_4_HOUR,
        "1d": CandleInterval.CANDLE_INTERVAL_DAY,
    }
    if interval not in m:
        raise InvestError(f"Неизвестный интервал: {interval}. Пример: 4h, 1h, 15m")
    return m[interval]


def fetch_candles(token: str, req: CandleRequest) -> list[Candle]:
    """Получаем свечи и возвращаем список dataclass Candle (Model)."""
    if Client is None or now is None:
        raise InvestError(
            "SDK T-Invest не установлен.\n"
            "Установите: pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple"
        )

    interval_enum = _interval_from_str(req.interval)
    from_dt = now() - timedelta(days=req.days_back)

    candles: list[Candle] = []

    try:
        with Client(token) as client:
            try:
                it = client.get_all_candles(
                    instrument_id=req.instrument_id,
                    interval=interval_enum,
                    from_=from_dt,
                )
            except TypeError:
                it = client.get_all_candles(
                    figi=req.instrument_id,
                    interval=interval_enum,
                    from_=from_dt,
                )

            for c in it:
                candles.append(
                    Candle(
                        time=c.time,
                        open=_quotation_to_float(c.open),
                        high=_quotation_to_float(c.high),
                        low=_quotation_to_float(c.low),
                        close=_quotation_to_float(c.close),
                        volume=int(getattr(c, "volume", 0)),
                    )
                )
    except Exception as e:
        raise InvestError(str(e)) from e

    if not candles:
        raise InvestError("Свечи не найдены (проверьте instrument_id/FIGI и период)")

    return candles


def candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
    df = pd.DataFrame([c.as_dict() for c in candles])
    df.set_index("time", inplace=True)
    return df


def plot_candles_base64(df: pd.DataFrame) -> str:
    """Строим свечной график и возвращаем PNG как base64 строку."""
    buf = BytesIO()
    mpf.plot(df, type="candle", volume=True, style="charles", savefig=buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def plot_volume_only_base64(df: pd.DataFrame) -> str:
    """
    ДОПОЛНИТЕЛЬНЫЙ ГРАФИК №1
    Строим отдельный график объема торгов и возвращаем PNG как base64 строку.
    """
    fig, ax = plt.subplots(figsize=(12, 4))


    colors = ['green' if close >= open_ else 'red'
              for close, open_ in zip(df['close'], df['open'])]

    ax.bar(df.index, df['volume'], color=colors, alpha=0.7, width=0.8)
    ax.set_title('Объем торгов', fontsize=14, fontweight='bold')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Объем')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close()

    return f"data:image/png;base64,{b64}"


def get_statistics(df: pd.DataFrame) -> dict:
    """
    ДОПОЛНИТЕЛЬНЫЙ ФУНКЦИОНАЛ - СТАТИСТИКА
    Рассчитывает основные статистические показатели по свечам.
    """

    first_close = df['close'].iloc[0]
    last_close = df['close'].iloc[-1]
    total_return = ((last_close - first_close) / first_close) * 100

    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * 100

    max_price = df['high'].max()
    min_price = df['low'].min()
    max_price_date = df['high'].idxmax().strftime('%Y-%m-%d %H:%M')
    min_price_date = df['low'].idxmin().strftime('%Y-%m-%d %H:%M')

    avg_volume = int(df['volume'].mean())

    avg_amplitude = ((df['high'] - df['low']) / df['low'] * 100).mean()

    return {
        'total_return': round(total_return, 2),
        'volatility': round(volatility, 2),
        'max_price': round(max_price, 2),
        'max_price_date': max_price_date,
        'min_price': round(min_price, 2),
        'min_price_date': min_price_date,
        'avg_volume': avg_volume,
        'avg_amplitude': round(avg_amplitude, 2),
        'num_candles': len(df),
        'first_date': df.index[0].strftime('%Y-%m-%d %H:%M'),
        'last_date': df.index[-1].strftime('%Y-%m-%d %H:%M'),
    }


def sdk_name() -> str:
    return _SDK_NAME