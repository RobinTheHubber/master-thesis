from tardis_dev import datasets, get_exchange_details
import logging

# optionally enable debug logs
# logging.basicConfig(level=logging.DEBUG)

exchange = 'deribit'
exchange_details = get_exchange_details(exchange)


btc = []
for symbol in exchange_details["datasets"]["symbols"]:
    if symbol['id'][:3] == 'BTC':
        btc.append(symbol['id'])
#TC-3DEC21-57000-C'.split('-')[
# # btc_in_date_selection = []
# # for btc_symbol in btc:
# #     if btc_symbol['']:
# #         date = 'B1]

# iterate over and download all data for every symbol
for symbol in exchange_details["datasets"]["symbols"]:
    # alternatively specify datatypes explicitly ['trades', 'incremental_book_L2', 'quotes'] etc
    # see available options https://docs.tardis.dev/downloadable-csv-files#data-types
    data_types = symbol["dataTypes"]
    symbol_id = symbol["id"]
    from_date =  symbol["availableSince"]
    to_date = symbol["availableTo"]

data_types = ['quotes']
symbol_id = 'BTC-31DEC21-26000-C'
from_date = '2021-12-15T00:00:00.000'
to_date = '2021-12-30T00:00:00.000'
    #
    # # skip groupped symbols
    # if symbol_id in ['PERPETUALS', 'SPOT', 'FUTURES']:
    #     continue

print(f"Downloading {exchange} {data_types} for {symbol_id} from {from_date} to {to_date}")

# each CSV dataset format is documented at https://docs.tardis.dev/downloadable-csv-files#data-types
# see https://docs.tardis.dev/downloadable-csv-files#download-via-client-libraries for full options docs
datasets.download(
    exchange = exchange,
    data_types = data_types,
    from_date = from_date,
    to_date = to_date,
    symbols = [symbol_id],
    # TODO set your API key here
    api_key = "TD.bOTVo8lO2QqbJwKE.LULDBGCpfzH2heE.x5Uz-inxkmv-UoF.eX7T8AJgwA6QQ6U.yRR6Ce35iS9ZGTv.tu9X",
    # path where CSV data will be downloaded into
    download_dir ="../datasets",
)

