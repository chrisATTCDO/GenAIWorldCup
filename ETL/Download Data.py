# Databricks notebook source
# MAGIC %md
# MAGIC # Download Data From Various Sources

# COMMAND ----------

import yfinance as yf

# COMMAND ----------

msft = yf.Ticker("MSFT")

# get all stock info
msft.info

# COMMAND ----------

msft.get_actions()

# COMMAND ----------

msft.get_calendar()

# COMMAND ----------

# import os
# import shutil
# shutil.rmtree(f"/dbfs/mnt/regression_testing1/hackathon_files/AT&T Inc")

# COMMAND ----------

# DBTITLE 1,.
import json
import os
from pandas import DataFrame
from datetime import date
from typing import List, Dict, Any

def default_converter(o: Any) -> str:
    """
    Convert date objects to ISO format for JSON serialization.
    """
    if isinstance(o, date):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

company = yf.Ticker("T")
company_name: str = company.info.get("longName")
company_symbol: str = company.info.get('symbol')
company_dir_path: str = f"/dbfs/mnt/regression_testing1/hackathon_files/{company_name}"

if not os.path.exists(company_dir_path):
    os.makedirs(company_dir_path)

functions: List[str] = [
    func for func in dir(company)
    if not func.startswith("_") and func not in ['shares'] and callable(getattr(company, func))
]

success_json: List[str] = []
dataframe_result: List[str] = []
proxy_error: List[Dict[str, str]] = []
other_error: List[Dict[str, str]] = []

for func in functions:
    try:
        result = getattr(company, func)()
        if isinstance(result, DataFrame):
            dataframe_result.append(func)
            result['symbol'] = company_symbol
            result.to_csv(f"{company_dir_path}/{func}.csv", header=True)
        elif isinstance(result, dict):
            result.update(symbol=company_symbol)
            with open(f'{company_dir_path}/{func}.json', 'w') as f:
                json.dump(result, f, default=default_converter)
            success_json.append(func)
    except Exception as e:
        error_message = str(e).lower()
        if "proxyerror" in error_message:
            proxy_error.append({'func': func, 'error': str(e)})
        else:
            other_error.append({'func': func, 'error': str(e)})

# COMMAND ----------

print(f"success jsons - {success_json} - total {len(success_json)}")
print(f"success csvs - {dataframe_result}, - total {len(dataframe_result)}")
print(f"errors - {other_error}, proxy-error - {proxy_error}")


# COMMAND ----------

data = spark.read.json("/mnt/regression_testing1/hackathon_files/AT&T Inc/get_analyst_price_targets.json")
# data = spark.read.csv("/mnt/regression_testing1/hackathon_files/AT&T Inc/get_recommendations.csv")
display(data)

# COMMAND ----------

# get historical market data
hist = msft.history(period="1mo")
hist.to_csv(f"/dbfs/mnt/regression_testing1/hackathon_files/hist.csv",header=True)
data = spark.read.csv("/mnt/regression_testing1/hackathon_files/hist.csv")
display(data)


# COMMAND ----------

# show meta information about the history (requires history() to be called first)
msft.history_metadata


# COMMAND ----------

# show actions (dividends, splits, capital gains)
msft.actions

# COMMAND ----------

msft.dividends

# COMMAND ----------


msft.splits

# COMMAND ----------

msft.capital_gains  # only for mutual funds & etfs


# COMMAND ----------

# show share count
msft.get_shares_full(start="2024-05-01", end=None)


# COMMAND ----------

# show financials:
msft.calendar

# COMMAND ----------

msft.sec_filings

# COMMAND ----------

# - income statement
msft.income_stmt

# COMMAND ----------

msft.quarterly_income_stmt


# COMMAND ----------

# - balance sheet
msft.balance_sheet

# COMMAND ----------

msft.quarterly_balance_sheet

# COMMAND ----------

# - cash flow statement
msft.cashflow

# COMMAND ----------

msft.quarterly_cashflow
# see `Ticker.get_income_stmt()` for more options


# COMMAND ----------

# show holders
msft.major_holders

# COMMAND ----------

msft.institutional_holders

# COMMAND ----------

msft.mutualfund_holders

# COMMAND ----------

msft.insider_transactions

# COMMAND ----------

msft.insider_purchases

# COMMAND ----------

msft.insider_roster_holders

# COMMAND ----------

msft.sustainability

# COMMAND ----------

# show recommendations
msft.recommendations

# COMMAND ----------

msft.recommendations_summary

# COMMAND ----------

msft.upgrades_downgrades

# COMMAND ----------

# show analysts data
msft.analyst_price_targets

# COMMAND ----------

msft.earnings_estimate

# COMMAND ----------

msft.revenue_estimate

# COMMAND ----------

msft.earnings_history

# COMMAND ----------

msft.eps_trend

# COMMAND ----------

msft.eps_revisions

# COMMAND ----------

msft.growth_estimates

# COMMAND ----------

# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
# Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
msft.earnings_dates


# COMMAND ----------

# # show ISIN code - *experimental*
# # ISIN = International Securities Identification Number
# msft.isin

# COMMAND ----------

# show options expirations
msft.options

# COMMAND ----------

# show news
msft.news

# COMMAND ----------

# get option chain for specific expiration
opt = msft.option_chain('2024-09-27')
# data available via: opt.calls, opt.puts
opt.calls

# COMMAND ----------

opt.puts

# COMMAND ----------


