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

  'Full-Time Employees': get_value(ticker.info, 'fullTimeEmployees'),
  'audit Risk': get_value(ticker.info, 'auditRisk'),
  'board Risk': get_value(ticker.info, 'boardRisk'),
  'compensation Risk': get_value(ticker.info, 'compensationRisk'),
  'shareholder Rights Risk': get_value(ticker.info, 'shareHolderRightsRisk'),
  'overall Risk': get_value(ticker.info, 'overallRisk'),

  # 'governance Date': epoch_to_date(ticker.info['governanceEpochDate']),
  # 'compensation As Of Date': epoch_to_date(ticker.info['compensationAsOfEpochDate']),

  'dividend Rate': get_value(ticker.info, 'dividendRate'),
  'dividend Yield': get_value(ticker.info, 'dividendYield'),
  'ex-Dividend Date': epoch_to_date(get_value(ticker.info, 'exDividendDate')),
  'payout Ratio': get_value(ticker.info, 'payoutRatio'),
  'five Year Average Dividend Yield': get_value(ticker.info, 'fiveYearAvgDividendYield'),
  'last Dividend Value': get_value(ticker.info, 'lastDividendValue'),
  'last Dividend Date': epoch_to_date(get_value(ticker.info, 'lastDividendDate')),
  'trailing Annual Dividend Rate': get_value(ticker.info, 'trailingAnnualDividendRate'),
  'trailing Annual Dividend Yield': get_value(ticker.info, 'trailingAnnualDividendYield'),

  'beta': get_value(ticker.info, 'beta'),
  'trailing PE': get_value(ticker.info, 'trailingPE'),
  'forward PE': get_value(ticker.info, 'forwardPE'),

  'currency': ticker.info['currency'],
  'market Cap': ticker.info['marketCap'],
  'enterprise Value': ticker.info['enterpriseValue'],

  'profit Margins': get_value(ticker.info, 'profitMargins'),

  'book Value': ticker.info['bookValue'],
  'price To Book': get_value(ticker.info, 'priceToBook'),
  'last Fiscal YearEnd': epoch_to_date(ticker.info['lastFiscalYearEnd']),
  'next Fiscal YearEnd': epoch_to_date(ticker.info['nextFiscalYearEnd']),
  'most Recent Quarter': epoch_to_date(ticker.info['mostRecentQuarter']),
  'earnings Quarterly Growth': get_value(ticker.info, 'earningsQuarterlyGrowth'),
  'net Income To Common': get_value(ticker.info, 'netIncomeToCommon'),
  'trailing Earnings Per Share (EPS)': get_value(ticker.info, 'trailingEps'),
  'forward Earnings Per Share (EPS)': get_value(ticker.info, 'forwardEps'),
  'peg Ratio': get_value(ticker.info, 'pegRatio'),
  'last Split Factor': get_value(ticker.info, 'lastSplitFactor'),
  'last Split Date': epoch_to_date(get_value(ticker.info, 'lastSplitDate')),
  'first Trade Date Utc': epoch_to_date(ticker.info['firstTradeDateEpochUtc']),

  'enterprise To Revenue': get_value(ticker.info, 'enterpriseToRevenue'),
  'enterprise To Ebitda': get_value(ticker.info, 'enterpriseToEbitda'),

  'total Cash': ticker.info['totalCash'],
  'total Cash Per Share': get_value(ticker.info, 'totalCashPerShare'),

  'ebitda': get_value(ticker.info, 'ebitda'),
  'total Debt': get_value(ticker.info, 'totalDebt'),
  'quick Ratio': get_value(ticker.info, 'quickRatio'),
  'current Ratio': get_value(ticker.info, 'currentRatio'),
  'total Revenue': get_value(ticker.info, 'totalRevenue'),
  'debt To Equity': get_value(ticker.info, 'debtToEquity'),

  'revenue Per Share': get_value(ticker.info, 'revenuePerShare'),
  'return On Assets': get_value(ticker.info, 'returnOnAssets'),
  'return On Equity': get_value(ticker.info, 'returnOnEquity'),
  'free Cashflow': get_value(ticker.info, 'freeCashflow'),

  'operating Cashflow': get_value(ticker.info, 'operatingCashflow'),
  'earnings Growth': get_value(ticker.info, 'earningsGrowth'),
  'revenue Growth': get_value(ticker.info, 'revenueGrowth'),
  'gross Margins': get_value(ticker.info, 'grossMargins'),
  'ebitda Margins': get_value(ticker.info, 'ebitdaMargins'),
  'operating Margins': get_value(ticker.info, 'operatingMargins'),

  'financial Currency': ticker.info['financialCurrency'],
  'trailing Peg Ratio': ticker.info['trailingPegRatio'],
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
  'average Volume': ticker.info['averageVolume'],
  'average Volume 10-days': ticker.info['averageVolume10days'],
  
  '52-Week Low': ticker.info['fiftyTwoWeekLow'],
  '52-Week High': ticker.info['fiftyTwoWeekHigh'],
  '50-Day Simple Moving Average': ticker.info['fiftyDayAverage'],
  '200-Day Simple Moving Average': ticker.info['twoHundredDayAverage'],
  '52-Week Change': ticker.info['52WeekChange'],
  'price To Sales Trailing 12 Months': get_value(ticker.info, 'priceToSalesTrailing12Months'),

  'float Shares': ticker.info['floatShares'],
  'shares Outstanding': ticker.info['sharesOutstanding'],
  'shares Short': ticker.info['sharesShort'],
  'shares Short Prior Month': ticker.info['sharesShortPriorMonth'],
  'shares Short Previous Month Date': ticker.info['sharesShortPreviousMonthDate'],
  'date Short Interest': epoch_to_date(ticker.info['dateShortInterest']),
  'shares Percent Shares Out': ticker.info['sharesPercentSharesOut'],

  'held Percent Insiders': ticker.info['heldPercentInsiders'],
  'held Percent Institutions': ticker.info['heldPercentInstitutions'],

  'short Ratio': ticker.info['shortRatio'],
  'short Percent Of Float': get_value(ticker.info, 'shortPercentOfFloat'),
  'implied Shares Outstanding': ticker.info['impliedSharesOutstanding'],
}

file_name = f"{File_Path}{Symbol}_fundamental.txt"
write_stock_info(file_name, company_header + "\nStock Fundamental in JSON Format", fundamental)
show_file_content(file_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Realtime Quote Placeholder
# MAGIC * Generate placeholder values... Realtime Job will overwrite this values

# COMMAND ----------

realtime = {
  'Current Stock Price': ticker.info['currentPrice'],
  'Previous Day Close': ticker.info['previousClose'],
  'day open Price': ticker.info['open'],
  'day Low Price': ticker.info['dayLow'],
  'day High Price': ticker.info['dayHigh'],
  'bid Size': get_value(ticker.info, 'bidSize'),
  'ask Size': get_value(ticker.info, 'askSize'),
  'Stock Volume': ticker.info['volume'],
}

file_name = f"{File_Path}{Symbol}_realtime.txt"
write_stock_info(file_name, company_header + "\nStock Realtime Quote JSON Format. Refresh every 15-minutes.", realtime)
show_file_content(file_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysts

# COMMAND ----------

try:
  df = ticker.upgrades_downgrades.reset_index()

  # Get List of Analysts
  mylist = []
  ana_list = []
  for index, row in df.loc[df['GradeDate'] > '2024-01-01'].iterrows():
    if len(mylist) > 9:
      break       # Enough count

    firm = row['Firm']
    
    if firm not in mylist:
      mylist.append(firm)
      ana_list.append({
        'Analyst Firm': firm, 
        'recommend rating': row['ToGrade'], 
        'rating date': str(row['GradeDate']), 
      })

  analyst = {
    'number Of Analyst Opinions': ticker.info['numberOfAnalystOpinions'],
    'recommendation': ticker.info['recommendationKey'],
    'recommended price': ticker.analyst_price_targets['current'],
    'recommended price low': ticker.analyst_price_targets['low'],
    'recommended price high': ticker.analyst_price_targets['high'],
    'recommended price mean': ticker.analyst_price_targets['mean'],
    'recommended price median': ticker.analyst_price_targets['median'],
    'recent analysts coverage': ana_list
  }

  file_name = f"{File_Path}{Symbol}_analyst.txt"
  write_stock_info(file_name, company_header + "\nanalysts and stock recommendation in JSON Format", analyst)
  show_file_content(file_name)
except Exception as e:
  print(e)

# COMMAND ----------

# ticker.upgrades_downgrades
# ticker.recommendations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Institution Holders

# COMMAND ----------

try:
  txt = str(ticker.major_holders) + "\nTop Institutions Holders\n" + str(ticker.institutional_holders) + "\nTop Mutual Funds Holders\n" + str(ticker.mutualfund_holders)

  file_name = f"{File_Path}{Symbol}_institutions_mutualfunds.txt"
  write_stock_info(file_name, company_header + "\nMajor Stock Holders\n" + txt, None)
  show_file_content(file_name)
except Exception as e:
  print(e)

# COMMAND ----------

# msft.institutional_holders
# msft.mutualfund_holders

# COMMAND ----------

# MAGIC %md
# MAGIC ### Insiders

# COMMAND ----------

try:
  txt = str(ticker.insider_purchases) + "\ninsider roster holders\n" + str(ticker.insider_roster_holders) + "\ninsider transactions\n" + str(ticker.insider_transactions)

  file_name = f"{File_Path}{Symbol}_insiders.txt"
  write_stock_info(file_name, company_header + "\ninsider purchases\n" + txt, None)
  show_file_content(file_name)
except Exception as e:
  print(e)

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

try:
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
except Exception as e:
  print(e)

# COMMAND ----------

print("Notebook Execution Completed:", datetime.datetime.now())

# COMMAND ----------

# display(dbutils.fs.ls("dbfs:/mnt/regression_testing/hackathon_files/pending"))

# COMMAND ----------


