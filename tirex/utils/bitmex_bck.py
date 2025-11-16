# -*- coding: utf-8 -*-
"""
BitMEX Authentication Module (backup version).

This module provides authentication strategies for BitMEX API requests.
"""
import os
import json
import warnings
import time
import numpy as np
import pandas as pd

import math
import hmac
import hashlib
import urllib
import requests
import uuid_utils as uuid
from pathlib import Path
from requests.auth import AuthBase

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

plt.ioff()

# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
week = mdates.WeekdayLocator()  # every week

# date format based on year
yearsFmt = mdates.DateFormatter('%Y')

np.random.seed(7)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    print("python-dotenv not installed, skipping .env loading.")

# BitMEX Configuration - Load from environment variables
BITMEX_API_KEY = os.getenv('BITMEX_API_KEY', '')
BITMEX_API_SECRET = os.getenv('BITMEX_API_SECRET', '')
BITMEX_URL = "wss://www.bitmex.com"

VERB = "GET"
ENDPOINT = "/realtime"


def generate_expires() -> int:
    return int(time.time() + 3600)


class AccessTokenAuth(AuthBase):
    """Attaches Access Token Authentication to the given Request object."""

    def __init__(self, accessToken):
        """Init with Token."""
        self.token = accessToken

    def __call__(self, r):
        """Called when forming a request - generates access token header."""
        if self.token:
            r.headers['access-token'] = self.token

        return r


def generate_signature(secret, verb, url, nonce, data):
    """Generate a request signature compatible with BitMEX."""
    # Parse the url so we can remove the base and extract just the path.

    parsedURL = urllib.parse.urlparse(url)
    path = parsedURL.path

    if parsedURL.query:
        path = path + '?' + parsedURL.query

    if isinstance(data, (bytes, bytearray)):
        data = data.decode('utf8')

    # print "Computing HMAC: %s" % verb + path + str(nonce) + data
    message = verb + path + str(nonce) + data

    signature = hmac.new(bytes(secret, 'utf8'), bytes(message, 'utf8'), digestmod=hashlib.sha256).hexdigest()

    return signature


class APIKeyAuth(AuthBase):
    """Attaches API Key Authentication to the given Request object."""

    def __init__(self, apiKey: str, apiSecret: str):
        """Init with Key & Secret."""
        self.apiKey = apiKey
        self.apiSecret = apiSecret

    def __call__(self, r):
        """Called when forming a request - generates api key headers."""
        # modify and return the request
        nonce = generate_expires()
        r.headers['api-expires'] = str(nonce)
        r.headers['api-key'] = self.apiKey
        r.headers['api-signature'] = generate_signature(self.apiSecret, r.method, r.url, nonce, r.body or '')

        return r


class APIKeyAuthWithExpires(AuthBase):
    """Attaches API Key Authentication to the given Request object. This implementation uses `expires`."""

    def __init__(self, apiKey: str, apiSecret: str):
        """Init with Key & Secret."""
        self.apiKey = apiKey
        self.apiSecret = apiSecret

    def __call__(self, r):
        """
        Called when forming a request - generates api key headers. This call uses `expires` instead of nonce.

        This way it will not collide with other processes using the same API Key if requests arrive out of order.
        For more details, see https://www.bitmex.com/app/apiKeys
        """
        # modify and return the request
        expires = int(round(time.time()) + 5)  # 5s
        # grace period in case of clock skew
        r.headers['api-expires'] = str(expires)
        r.headers['api-key'] = self.apiKey
        r.headers['api-signature'] = generate_signature(self.apiSecret, r.method, r.url, expires, r.body or '')

        return r


class BitMEX:
    __name__ = 'BitMEX'

    """BitMEX API Connector with improved error handling and refactored code."""

    def __init__(self,
                 base_url: str = None,
                 symbol: str = None,
                 login=None,
                 password: str = None,
                 otpToken=None,
                 orderIDPrefix: str = 'mm_bitmex_',
                 ):

        """Initialize connector."""
        if len(orderIDPrefix) > 13:
            raise ValueError("ORDERID_PREFIX must be at most 13 characters long!")

        self.base_url = base_url
        self.symbol = symbol
        self.token = None
        self.login = login
        self.password = password
        self.otpToken = otpToken
        self.apiKey = BITMEX_API_KEY
        self.apiSecret = BITMEX_API_SECRET

        self.orderIDPrefix = orderIDPrefix
        self.session = self._initialize_session()

    @staticmethod
    def _initialize_session():
        """Initialize the HTTP session with default headers."""

        # Prepare HTTPS session
        session = requests.Session()

        # These headers are always sent
        session.headers.update({'user-agent': 'easy-data-scripts'})
        return session

    # Public methods
    def ticker_data(self):
        """Get ticker data."""
        data = self.get_instrument()

        ticker = {
            # Rounding to tickLog covers up float error
            "last": data['lastPrice'],
            "buy": data['bidPrice'],
            "sell": data['askPrice'],
            "mid": (float(data['bidPrice']) + float(data['askPrice'])) / 2
        }

        return {k: round(float(v), data['tickLog']) for k, v in ticker.items()}

    def get_instrument(self):
        """Get an instrument's details."""
        path = "instrument"
        instruments = self._curl_bitmex(path=path, query={'filter': json.dumps({'symbol': self.symbol})})
        if len(instruments) == 0:
            print("Instrument not found: %s." % self.symbol)
            exit(1)

        instrument = instruments[0]
        if instrument["state"] != "Open":
            print("The instrument %s is no longer open. State: %s" % (self.symbol, instrument["state"]))
            exit(1)

        # tickLog is the log10 of tickSize
        instrument['tickLog'] = int(math.fabs(math.log10(instrument['tickSize'])))

        return instrument

    def market_depth(self):
        """Get market depth / orderbook."""
        path = "orderBook"
        return self._curl_bitmex(path=path, query={'symbol': self.symbol})

    def recent_trades(self):
        """Get recent trades.

        Returns
        -------
        A list of dicts:
              {u'amount': 60,
               u'date': 1306775375,
               u'price': 8.7401099999999996,
               u'tid': u'93842'},

        """
        path = "trade"
        return self._curl_bitmex(path=path)

    @property
    def snapshot(self):
        """Get current BBO."""
        order_book = self.market_depth()
        return {
            'bid': order_book[0]['bidPrice'],
            'ask': order_book[0]['askPrice'],
            'size_bid': order_book[0]['bidSize'],
            'size_ask': order_book[0]['askSize']
        }

    # Authentication required methods
    def authenticate(self):
        """Set BitMEX authentication information."""
        if self.apiKey:
            return
        login_response = self._curl_bitmex(
            path="user/login",
            postdict={'email': self.login, 'password': self.password, 'token': self.otpToken})
        self.token = login_response['id']
        self.session.headers.update({'access-token': self.token})

    def authentication_required(fn):
        """Annotation for methods that require auth."""

        def wrapped(self, *args, **kwargs):
            if not (self.token or self.apiKey):
                msg = "You must be authenticated to use this method"
                raise msg
            else:
                return fn(self, *args, **kwargs)

        return wrapped

    @authentication_required
    def funds(self):
        """Get your current balance."""
        return self._curl_bitmex(path="user/margin")

    @authentication_required
    def buy(self, quantity: float, price: float):
        """Place a buy order.

        Returns order object. ID: orderID
        """
        return self.place_order(quantity, price)

    @authentication_required
    def sell(self, quantity: float, price: float):
        """Place a sell order.

        Returns order object. ID: orderID
        """
        return self.place_order(-quantity, price)

    @authentication_required
    def place_order(self, quantity: float, price: float):
        """Place an order."""
        if price < 0:
            raise Exception("Price must be positive.")

        endpoint = "order"
        # Generate a unique clOrdID with our prefix so we can identify it.
        # FIXME
        clOrdID = self.orderIDPrefix + uuid.uuid4().bytes.encode('base64').rstrip('=\n')
        postdict = {
            'symbol': self.symbol,
            'quantity': quantity,
            'price': price,
            'clOrdID': clOrdID
        }
        return self._curl_bitmex(path=endpoint, postdict=postdict, verb="POST")

    @authentication_required
    def open_orders(self):
        """Get open orders."""
        path = "order"

        filter_dict = {'ordStatus.isTerminated': False}
        if self.symbol:
            filter_dict['symbol'] = self.symbol

        orders = self._curl_bitmex(
            path=path,
            query={'filter': json.dumps(filter_dict)},
            verb="GET"
        )
        # Only return orders that start with our clOrdID prefix.
        return [o for o in orders if str(o['clOrdID']).startswith(self.orderIDPrefix)]

    @authentication_required
    def cancel(self, orderID: int):
        """Cancel an existing order."""

        path = "order"
        postdict = {'orderID': orderID}

        return self._curl_bitmex(path=path, postdict=postdict, verb="DELETE")

    def get_net_chart(self, hours: float, cpair: str, fn_time: pd.Timestamp = None, dt: int = 1) -> pd.DataFrame:

        print(' Loading bitmex chart...')

        count = 700

        if dt == 1:
            str_dt = '5m'
            dt_i = 5

        elif dt == 3:
            # 15 minutes ticker
            # str_dt = '15m'
            str_dt = '5m'
            dt_i = 5

        elif dt == 6:
            # 30 minutes ticker
            # str_dt = '15m'
            str_dt = '5m'
            dt_i = 5

        elif dt == 12:
            # 60 minutes ticker
            str_dt = '1h'
            dt_i = 60

        query = {'start': 0, 'count': count, 'symbol': cpair, 'binSize': str_dt, 'reverse': False}

        if fn_time is None:
            # fn_time = pd.Timestamp.now() + pd.Timedelta(hours=3)
            fn_time = pd.Timestamp.now() + pd.Timedelta(hours=1)

        delta_time = pd.Timedelta(hours=hours)

        st_time = fn_time - delta_time

        dt_min = pd.Timedelta(hours=hours) / pd.Timedelta(minutes=dt_i)

        # dt_loop = dt_min / count
        dt_loop = dt_min // count

        list_loop_dtime = []
        list_loop_start_data = []
        list_loop_end_data = []

        list_open = []
        list_close = []
        list_high = []
        list_low = []
        list_time = []
        list_ntrades = []
        list_vol = []

        if dt_loop < 1.:
            num_loop = 1
        else:
            num_loop = int(dt_loop) + 1

        for i in tqdm(range(num_loop)):

            if dt_loop > 1.:
                dtime_i = count * pd.Timedelta(minutes=dt_i)

            elif 0. < dt_loop <= 1.:
                dtime_i = dt_loop * count * pd.Timedelta(minutes=dt_i)

            elif dt_loop <= 0:
                dtime_i = dt_min * pd.Timedelta(minutes=dt_i)

            list_loop_dtime.append(dtime_i)

            if i == 0:
                start_time_i = st_time

            query['startTime'] = start_time_i.strftime('%Y-%m-%dT%H:%M:%S')
            list_loop_start_data.append(start_time_i)

            start_time_i += dtime_i

            query['endTime'] = start_time_i.strftime('%Y-%m-%dT%H:%M:%S')
            list_loop_end_data.append(start_time_i)

            for w in range(10):
                for k in range(20):
                    try:
                        list_data = self._curl_bitmex(path='trade/bucketed', verb="GET", query=query, timeout=10)
                        break
                    except:
                        print(' Trying again ' + str(k))
                        # time.sleep(0.2)
                        time.sleep(1.2)
                        list_data = None

                if list_data is None:
                    print('\n Error, It is not possible to get data from BitMEX, trying again in 70 seconds... \n')
                    time.sleep(70.)
                else:
                    break

            if list_data is None:
                raise ValueError('BitMEX WebSocket is not accessible')

            try:
                for data_i in reversed(list_data):
                    list_open.append(data_i['open'])
                    list_close.append(data_i['close'])
                    list_high.append(data_i['high'])
                    list_low.append(data_i['low'])
                    list_time.append(pd.to_datetime(data_i['timestamp'], errors='coerce'))
                    list_ntrades.append(data_i['trades'])
                    list_vol.append(data_i['volume'])
            except:
                # FIXME
                print(' ERRO!!!')

            time.sleep(0.5)
            dt_loop -= 1.

        list_keys = ['open', 'close', 'high', 'low', 'trades', 'volume']
        list_value = [list_open, list_close, list_high, list_low, list_ntrades, list_vol]

        dict_data = {key_i: np.array(val_i, dtype=np.float32) for (key_i, val_i) in zip(list_keys, list_value)}

        # Create NumPy array without timezone information
        dict_data['date'] = np.array([time_k.replace(tzinfo=None) for time_k in list_time], dtype='datetime64[s]')

        dict_data_ou = dict_data.copy()
        dict_data_ou.pop('date')

        pd_crypto_data = pd.DataFrame(dict_data_ou)
        pd_crypto_data.index = pd.to_datetime(dict_data['date'])

        if dt != 3 and dt != 6:
            return pd_crypto_data

        elif dt == 3:
            dict_values_dt3 = {k_i: [] for k_i, v_i in dict_data.items()}
            np_dt3_inc = np.array([0, 15, 30, 45], dtype=int)

            ii = 0
            for idx_i in range(dict_data['date'].size):
                data_i = dict_data['date'][idx_i].item().minute

                if (data_i in np_dt3_inc) is True and ii > 2:
                    np_idx_i = np.arange(idx_i - 3, idx_i) + 1

                    dict_values_dt3['date'].append(dict_data['date'][np_idx_i][-1])

                    dict_values_dt3['open'].append(dict_data['open'][np_idx_i][0])
                    dict_values_dt3['close'].append(dict_data['close'][np_idx_i][-1])
                    dict_values_dt3['high'].append(dict_data['high'][np_idx_i].max())
                    dict_values_dt3['low'].append(dict_data['low'][np_idx_i].min())
                    dict_values_dt3['volume'].append(dict_data['volume'][np_idx_i].sum())
                    dict_values_dt3['trades'].append(dict_data['trades'][np_idx_i].sum())

                    ii = 1

                else:
                    ii += 1

            # Array correction
            dict_data_ou = {k_i: np.array(v_i) for k_i, v_i in dict_values_dt3.items()}

            dict_data_pd = dict_data_ou.copy()
            dict_data_pd.pop('date')

            pd_crypto_data = pd.DataFrame(dict_data_pd)
            pd_crypto_data.index = pd.to_datetime(dict_data_ou['date'])

            return pd_crypto_data

        elif dt == 6:
            dict_values_dt6 = {k_i: [] for k_i, v_i in dict_data.items()}
            dict_data_ex = {k_i: dict_data[k_i] for k_i in ['open', 'close', 'high', 'low', 'trades', 'volume']}
            pd_data = pd.DataFrame(dict_data_ex, index=dict_data['date'])
            pd_data.dropna(inplace=True)

            np_dt6_inc = np.array([0, 30], dtype=int)

            np_min_idx = np.array([idx.minute for idx in pd_data.index])

            list_idx = []
            for dt6_i in np_dt6_inc:
                list_idx.append(np.where(np_min_idx == dt6_i)[0])

            np_slc_idx = np.concatenate(list_idx)
            np_slc_idx.sort()

            np_range_idx = np_slc_idx[10:]

            for i, idx_i in enumerate(np_range_idx[:-1]):
                np_idx_i = np.arange(idx_i, np_range_idx[i + 1]) + 1

                pd_data_i = pd_data.iloc[np_idx_i, :]

                dict_values_dt6['date'].append(pd_data_i.index[-1])

                dict_values_dt6['open'].append(pd_data_i['open'][0])
                dict_values_dt6['close'].append(pd_data_i['close'][-1])
                dict_values_dt6['high'].append(pd_data_i['high'].max())
                dict_values_dt6['low'].append(pd_data_i['low'].min())
                dict_values_dt6['volume'].append(pd_data_i['volume'].sum())
                dict_values_dt6['trades'].append(pd_data_i['trades'].sum())

            # Array correction
            dict_data_ou = {k_i: np.array(v_i) for k_i, v_i in dict_values_dt6.items()}

            dict_data_pd = dict_data_ou.copy()
            dict_data_pd.pop('date')

            pd_crypto_data = pd.DataFrame(dict_data_pd)
            pd_crypto_data.index = pd.to_datetime(dict_data_ou['date'])

            return pd_crypto_data

    def _handle_http_error(self, response, verb, path, postdict):
        """Handle HTTP error codes and implement retry logic."""
        if response.status_code == 401:
            self._reauthenticate()
        elif response.status_code == 429:
            time.sleep(25)
        elif response.status_code == 503:
            time.sleep(1)
        else:
            raise Exception(f"HTTP Error {response.status_code}: {response.text}")

    def _reauthenticate(self):
        """Re-authenticate when token expires or is invalid."""
        print("Token expired, reauthenticating...")
        time.sleep(1)
        self.authenticate()


    def _curl_bitmex(self, path: str, query=None, postdict=None, timeout=3, verb=None):
        """Send a request to BitMEX Servers."""

        # Handle URL
        url = self.base_url + path

        # Default to POST if data is attached, GET otherwise
        verb = verb or 'POST' if postdict else 'GET'

        # Auth: Use Access Token by default, API Key/Secret if provided
        auth = AccessTokenAuth(self.token) if self.apiKey is None else APIKeyAuthWithExpires(self.apiKey, self.apiSecret)

        # Make the request
        for attempt in range(3):  # Retry logic
            try:
                req = requests.Request(verb, url, data=postdict, auth=auth, params=query)
                prepped = self.session.prepare_request(req)
                response = self.session.send(prepped, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                self._handle_http_error(response, verb, path, postdict)
            except requests.exceptions.Timeout:
                if attempt == 2:  # Only retry twice
                    raise Exception(f"Request timed out after {attempt + 1} attempts.")
                time.sleep(1)  # Retry after a delay
            except requests.exceptions.ConnectionError:
                if attempt == 2:
                    raise Exception("Unable to connect after retries.")
                time.sleep(1)


class GetDataPair:

    def __init__(self):
        self.obj_data_trade = BitMEX(base_url='https://www.bitmex.com/api/v1/')

    def __call__(self, cpair: str, hours: float, dt: int, fn_time: pd.Timestamp = None) -> pd.DataFrame:

        pd_chart_out = pd.DataFrame()

        if cpair == 'XBTUSD' or cpair == 'BTCUSD':
            pd_chart_out = self.obj_data_trade.get_net_chart(hours, 'XBTUSD', fn_time=fn_time, dt=dt)

        elif cpair == 'ETHUSD':
            pd_chart_out = self.obj_data_trade.get_net_chart(hours, 'ETHUSD', fn_time=fn_time, dt=dt)

        elif cpair == 'LTCUSD':
            pd_chart_out = self.obj_data_trade.get_net_chart(hours, 'LTCUSD', fn_time=fn_time, dt=dt)

        elif cpair == 'XRPUSD':
            pd_chart_out = self.obj_data_trade.get_net_chart(hours, 'XRPUSD', fn_time=fn_time, dt=dt)

        return pd_chart_out


def save_ticker(dt: int = 60, size: int = 80000):

    current_file_path = Path(__file__).parent.parent.parent
    parquete_path = current_file_path / "Signals/data"
    parquete_path.mkdir(parents=True, exist_ok=True)

    get_data_pair = GetDataPair()

    pd_time = pd.Timestamp.now()
    dt_inc = pd_time.minute // dt

    end_date = pd.Timestamp(year=pd_time.year, month=pd_time.month, day=pd_time.day, hour=pd_time.hour, minute=dt_inc * dt)

    nhours = 1000
    # nhours = 80000
    # nhours = 20000
    fn_name = f"btcusd_{dt}m_{end_date.strftime('%Y-%m-%d')}"

    # pd_chart = get_data_pair('XBTUSD', nhours, dt // 5, fn_time=end_date)
    pd_chart = get_data_pair('XBTUSD', nhours, dt // 5)
    pd_chart = pd_chart[-size:]
    pd_chart.to_parquet(f"{parquete_path}/{fn_name}.parquet")


if __name__ == '__main__':
    save_ticker(dt=15)
