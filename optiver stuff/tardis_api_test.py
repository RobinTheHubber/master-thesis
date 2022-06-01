import pandas as pd
from tardis_dev import datasets, get_exchange_details

exchange = 'deribit'
exchange_details = get_exchange_details(exchange)

def date_in_range(symbol):
    since = symbol['availableSince']
    to = symbol['availableTo']

    # year, month, day = since.split('-')
    # day = day[:2]
    # if not (year == '2022' and month == '05' and day >= '13'):
    #     return False

    year, month, day = to.split('-')
    day = day[:2]

    if not (year == '2022' and month == '05' and day <= '28'):
        return False

    return True

btc_symbols = []
for symbol in exchange_details["datasets"]["symbols"]:
    symbol_id = symbol['id']
    if symbol_id[:3] == 'BTC' and symbol['type']=='option' \
            and 'quotes' in symbol['dataTypes'] and date_in_range(symbol) \
            and symbol['stats']['trades'] > 1000:
        btc_symbols.append(symbol)


## RUNTIME WARNING: takes like 5-10 minutes on my end
# iterate over and download all data for every symbol
i = 0
for symbol in btc_symbols:
    # alternatively specify datatypes explicitly ['trades', 'incremental_book_L2', 'quotes'] etc
    # see available options https://docs.tardis.dev/downloadable-csv-files#data-types
    data_types = ['quotes']
    symbol_id = symbol["id"]
    from_date = symbol["availableSince"]

    year, month, day = from_date.split('-')
    day = day[:2]

    if year < '2022' or month < '05' or day < '13':
        from_date = '2022-05-13T00:00:00.000Z'

    to_date = symbol["availableTo"]

    i += 1
    print(f"run {i} Downloading {exchange} {data_types} for {symbol_id} from {from_date} to {to_date}")

    # each CSV dataset format is documented at https://docs.tardis.dev/downloadable-csv-files#data-types
    # see https://docs.tardis.dev/downloadable-csv-files#download-via-client-libraries for full options docs
    datasets.download(
        exchange = exchange,
        data_types = data_types,
        from_date = from_date,
        to_date = to_date,
        symbols = [symbol_id],
        api_key = "TD.bOTVo8lO2QqbJwKE.LULDBGCpfzH2heE.x5Uz-inxkmv-UoF.eX7T8AJgwA6QQ6U.yRR6Ce35iS9ZGTv.tu9X",
        # path where CSV data will be downloaded into
        download_dir ="../deribit_/"+ symbol_id,
    )

## example of how to load in the quote data for a single day for a given call option
# note parameter nrow indicates that you only want to load in first nrow number of rows
data = pd.read_csv('datasets/deribit_/BTC-27MAY22-80000-C/deribit_quotes_2022-05-26_BTC-27MAY22-80000-C.csv.gz', compression='gzip')
