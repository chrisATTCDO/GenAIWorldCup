# Databricks notebook source
# MAGIC %md
# MAGIC # Download Data From Various Sources Per Stock Symbol

# COMMAND ----------

dbutils.widgets.text("Stock_Symbol", "T")

# COMMAND ----------

import yfinance as yf
import json
from pprint import pprint
from time import strftime, localtime
import datetime

# COMMAND ----------

Symbol = dbutils.widgets.get("Stock_Symbol").strip().upper()

if Symbol == "":
  dbutils.notebook.exit("Stock Symbol is required!")

print("Symbol:", Symbol)

File_Path = '/dbfs/mnt/regression_testing/hackathon_files/pending/'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions

# COMMAND ----------

def get_value(obj, key):
  try:
    return obj[key]
  except:
    return ''

# COMMAND ----------

def epoch_to_date(epoch_time):
  if epoch_time == '':
      return ''
    
  return strftime('%m/%d/%Y %H:%M:%S', localtime(epoch_time))

# COMMAND ----------

def write_stock_info(file_name, header, my_dict):
  with open(file_name, 'w') as file:
    file.write(header + "\n")
    file.write(json.dumps(my_dict)) 


# COMMAND ----------

def show_file_content(filename, MAX_LINES=5):
  count = 1
  
  with open(filename, "r") as f:
    lines = f.readlines()
    print(f"{filename}\tLines: {len(lines)}\n========================================================================================")
    for line in lines:
      print(line) 

      count += 1
      if count > MAX_LINES:
        break

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Data

# COMMAND ----------

ticker = yf.Ticker(Symbol)

# COMMAND ----------

company_header = "Company: " + ticker.info['longName'] + "\nStock Symbol: " + Symbol

print(company_header)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Company Profile

# COMMAND ----------

profile = {
  'industry': ticker.info['industry'],
  'sector': ticker.info['sector'],
  'Business Summary': ticker.info['longBusinessSummary'],
  'Stock Exchange': ticker.info['exchange'],
  'Quote Type': ticker.info['quoteType'],

  'Full-Time Employees': ticker.info['fullTimeEmployees'],
  'auditRisk': ticker.info['auditRisk'],
  'boardRisk': ticker.info['boardRisk'],
  'compensationRisk': ticker.info['compensationRisk'],
  'shareHolderRightsRisk': ticker.info['shareHolderRightsRisk'],
  'overallRisk': ticker.info['overallRisk'],

  'governanceEpochDate': epoch_to_date(ticker.info['governanceEpochDate']),
  'compensationAsOfEpochDate': epoch_to_date(ticker.info['compensationAsOfEpochDate']),

  'dividendRate': get_value(ticker.info, 'dividendRate'),
  'dividendYield': get_value(ticker.info, 'dividendYield'),
  'exDividendDate': epoch_to_date(get_value(ticker.info, 'exDividendDate')),
  'payoutRatio': get_value(ticker.info, 'payoutRatio'),
  'fiveYearAvgDividendYield': get_value(ticker.info, 'fiveYearAvgDividendYield'),
  'lastDividendValue': get_value(ticker.info, 'lastDividendValue'),
  'lastDividendDate': epoch_to_date(get_value(ticker.info, 'lastDividendDate')),
  'trailingAnnualDividendRate': get_value(ticker.info, 'trailingAnnualDividendRate'),
  'trailingAnnualDividendYield': get_value(ticker.info, 'trailingAnnualDividendYield'),

  'beta': ticker.info['beta'],
  'trailingPE': ticker.info['trailingPE'],
  'forwardPE': ticker.info['forwardPE'],

  'currency': ticker.info['currency'],
  'marketCap': ticker.info['marketCap'],
  'enterpriseValue': ticker.info['enterpriseValue'],

  'profitMargins': ticker.info['profitMargins'],

  'bookValue': ticker.info['bookValue'],
  'priceToBook': ticker.info['priceToBook'],
  'lastFiscalYearEnd': epoch_to_date(ticker.info['lastFiscalYearEnd']),
  'nextFiscalYearEnd': epoch_to_date(ticker.info['nextFiscalYearEnd']),
  'mostRecentQuarter': epoch_to_date(ticker.info['mostRecentQuarter']),
  'earningsQuarterlyGrowth': ticker.info['earningsQuarterlyGrowth'],
  'netIncomeToCommon': ticker.info['netIncomeToCommon'],
  'trailingEps': ticker.info['trailingEps'],
  'forwardEps': ticker.info['forwardEps'],
  'pegRatio': ticker.info['pegRatio'],
  'lastSplitFactor': ticker.info['lastSplitFactor'],
  'lastSplitDate': epoch_to_date(ticker.info['lastSplitDate']),

  'enterpriseToRevenue': ticker.info['enterpriseToRevenue'],
  'enterpriseToEbitda': ticker.info['enterpriseToEbitda'],

  'firstTradeDateEpochUtc': epoch_to_date(ticker.info['firstTradeDateEpochUtc']),

  'totalCash': ticker.info['totalCash'],
  'totalCashPerShare': ticker.info['totalCashPerShare'],

  'ebitda': ticker.info['ebitda'],
  'totalDebt': ticker.info['totalDebt'],
  'quickRatio': ticker.info['quickRatio'],
  'currentRatio': ticker.info['currentRatio'],
  'totalRevenue': ticker.info['totalRevenue'],
  'debtToEquity': ticker.info['debtToEquity'],

  'revenuePerShare': ticker.info['revenuePerShare'],
  'returnOnAssets': ticker.info['returnOnAssets'],
  'returnOnEquity': ticker.info['returnOnEquity'],
  'freeCashflow': ticker.info['freeCashflow'],

  'operatingCashflow': ticker.info['operatingCashflow'],
  'earningsGrowth': ticker.info['earningsGrowth'],
  'revenueGrowth': ticker.info['revenueGrowth'],
  'grossMargins': ticker.info['grossMargins'],
  'ebitdaMargins': ticker.info['ebitdaMargins'],
  'operatingMargins': ticker.info['operatingMargins'],

  'financialCurrency': ticker.info['financialCurrency'],
  'trailingPegRatio': ticker.info['trailingPegRatio'],
}

file_name = f"{File_Path}{Symbol}_profile.txt"
write_stock_info(file_name, company_header + "\nStock Profile in JSON Format", profile)
show_file_content(file_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Contact

# COMMAND ----------

contact = {
  'address1': ticker.info['address1'],
  'city': ticker.info['city'],
  'state': ticker.info['state'],
  'zip': ticker.info['zip'],
  'country': ticker.info['country'],
  'phone': ticker.info['phone'],
  'website': ticker.info['website'],
  'Investor Relation (IR) Website': get_value(ticker.info, 'irWebsite'),
}

file_name = f"{File_Path}{Symbol}_contact.txt"
write_stock_info(file_name, company_header + "\nContact in JSON Format", contact)
show_file_content(file_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Company Officers

# COMMAND ----------

companyOfficers = ticker.info['companyOfficers']

file_name = f"{File_Path}{Symbol}_companyOfficers.txt"
write_stock_info(file_name, company_header + "\nCompany Officers in JSON Format", companyOfficers)
show_file_content(file_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fundamental

# COMMAND ----------

fundamental = {
  'Previous Stock Close': ticker.info['previousClose'],
  'Stock open': ticker.info['open'],
  'Stock day Low': ticker.info['dayLow'],
  'Stock day High': ticker.info['dayHigh'],
  'volume': ticker.info['volume'],
  'averageVolume': ticker.info['averageVolume'],
  'averageVolume10days': ticker.info['averageVolume10days'],
  'bidSize': ticker.info['bidSize'],
  'askSize': ticker.info['askSize'],

  'fiftyTwoWeekLow': ticker.info['fiftyTwoWeekLow'],
  'fiftyTwoWeekHigh': ticker.info['fiftyTwoWeekHigh'],
  'fiftyDayAverage': ticker.info['fiftyDayAverage'],
  'twoHundredDayAverage': ticker.info['twoHundredDayAverage'],
  '52WeekChange': ticker.info['52WeekChange'],
  'priceToSalesTrailing12Months': ticker.info['priceToSalesTrailing12Months'],

  'floatShares': ticker.info['floatShares'],
  'sharesOutstanding': ticker.info['sharesOutstanding'],
  'sharesShort': ticker.info['sharesShort'],
  'sharesShortPriorMonth': ticker.info['sharesShortPriorMonth'],
  'sharesShortPreviousMonthDate': ticker.info['sharesShortPreviousMonthDate'],
  'dateShortInterest': epoch_to_date(ticker.info['dateShortInterest']),
  'sharesPercentSharesOut': ticker.info['sharesPercentSharesOut'],

  'heldPercentInsiders': ticker.info['heldPercentInsiders'],
  'heldPercentInstitutions': ticker.info['heldPercentInstitutions'],

  'shortRatio': ticker.info['shortRatio'],
  'shortPercentOfFloat': ticker.info['shortPercentOfFloat'],
  'impliedSharesOutstanding': ticker.info['impliedSharesOutstanding'],

  'currentPrice': ticker.info['currentPrice'],
  'targetHighPrice': ticker.info['targetHighPrice'],
  'targetLowPrice': ticker.info['targetLowPrice'],
  'targetMeanPrice': ticker.info['targetMeanPrice'],
  'targetMedianPrice': ticker.info['targetMedianPrice'],
}

file_name = f"{File_Path}{Symbol}_fundamental.txt"
write_stock_info(file_name, company_header + "\nStock Fundamental in JSON Format", fundamental)
show_file_content(file_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysts

# COMMAND ----------

analyst = {
  'numberOfAnalystOpinions': ticker.info['numberOfAnalystOpinions'],
  'recommendationKey': ticker.info['recommendationKey'],
  'current': ticker.analyst_price_targets['current'],
  'low': ticker.analyst_price_targets['low'],
  'high': ticker.analyst_price_targets['high'],
  'mean': ticker.analyst_price_targets['mean'],
  'median': ticker.analyst_price_targets['median'],
}

file_name = f"{File_Path}{Symbol}_analyst.txt"
write_stock_info(file_name, company_header + "\nanalysts and stock recommendation in JSON Format", analyst)
show_file_content(file_name)

# COMMAND ----------

# ticker.upgrades_downgrades
# ticker.recommendations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Institution Holders

# COMMAND ----------

txt = str(ticker.major_holders) + "\nTop Institutions Holders\n" + str(ticker.institutional_holders) + "\nTop Mutual Funds Holders\n" + str(ticker.mutualfund_holders)

file_name = f"{File_Path}{Symbol}_institutions_mutualfunds.txt"
write_stock_info(file_name, company_header + "\nMajor Stock Holders\n" + txt, None)
show_file_content(file_name)

# COMMAND ----------

# msft.institutional_holders
# msft.mutualfund_holders

# COMMAND ----------

# MAGIC %md
# MAGIC ### Insiders

# COMMAND ----------

txt = str(ticker.insider_purchases) + "\ninsider roster holders\n" + str(ticker.insider_roster_holders) + "\ninsider transactions\n" + str(ticker.insider_transactions)

file_name = f"{File_Path}{Symbol}_insiders.txt"
write_stock_info(file_name, company_header + "\ninsider purchases\n" + txt, None)
show_file_content(file_name)

# COMMAND ----------

# msft.insider_transactions
# msft.insider_purchases
# msft.insider_roster_holders

# COMMAND ----------

# MAGIC %md
# MAGIC ### Earnings

# COMMAND ----------

# msft.earnings_estimate
# msft.revenue_estimate
# msft.earnings_history
# msft.eps_trend
# msft.eps_revisions
# msft.growth_estimates

# COMMAND ----------

# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
# Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
# msft.earnings_dates


# COMMAND ----------

# # show ISIN code - *experimental*
# # ISIN = International Securities Identification Number
# msft.isin

# COMMAND ----------

# MAGIC %md
# MAGIC ### Options

# COMMAND ----------

# show options expirations
# msft.options

# COMMAND ----------

# # get option chain for specific expiration
# opt = msft.option_chain('2024-09-27')
# # data available via: opt.calls, opt.puts
# opt.calls

# COMMAND ----------

# opt.puts

# COMMAND ----------

# MAGIC %md
# MAGIC ### News

# COMMAND ----------

news = []

for item in ticker.news:
  news.append({
    'News Title': item['title'],
    'Link to News Article': item['link'],
    'Publisher': item['publisher'],
    'Provider Publish Time': epoch_to_date(item['providerPublishTime']),
    'News Type': item['type'],
    'Related Tickers': item['relatedTickers'],
  })

file_name = f"{File_Path}{Symbol}_news.txt"
write_stock_info(file_name, company_header + "\nNews Articles in JSON Format", news)
show_file_content(file_name)

# COMMAND ----------

print("Notebook Execution Completed:", datetime.datetime.now())

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/regression_testing/hackathon_files/pending"))

# COMMAND ----------


