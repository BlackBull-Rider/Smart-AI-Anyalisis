from __future__ import annotations

import pandas as pd

from backend.providers.base_provider import BaseProvider


class NSEProvider(BaseProvider):
    """
    NSE Universe Provider
    """

    def get_universe(self) -> pd.DataFrame:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

        df = pd.read_csv(url)

        df = df.rename(
            columns={
                "SYMBOL": "symbol",
                "NAME OF COMPANY": "company_name",
            }
        )

        df = df[["symbol", "company_name"]].copy()
        df["exchange"] = "NSE"

        df["symbol"] = df["symbol"].astype(str).str.strip()
        df["company_name"] = df["company_name"].astype(str).str.strip()

        return df

    def get_history(self, *args, **kwargs):
        raise NotImplementedError

    def get_company_info(self, *args, **kwargs):
        raise NotImplementedError

    def get_fundamentals(self, *args, **kwargs):
        raise NotImplementedError

    def get_financials(self, *args, **kwargs):
        raise NotImplementedError

    def get_actions(self, *args, **kwargs):
        raise NotImplementedError

    def get_share_holders(self, *args, **kwargs):
        raise NotImplementedError

    def get_earnings(self, *args, **kwargs):
        raise NotImplementedError

    def get_recommendations(self, *args, **kwargs):
        raise NotImplementedError
