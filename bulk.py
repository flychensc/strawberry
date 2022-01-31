from rqalpha.apis import *
from sklearn.utils import shuffle

import datetime as dt
import mplfinance as mpf
import numpy as np
import pandas as pd

import configparser
import os


def gen_kline(context, data, fpath):
    data = pd.DataFrame(data)
    data['datetime'] = data['datetime'].map(lambda x: dt.datetime.strptime(str(x), "%Y%m%d%H%M%S").date())
    data.index = pd.to_datetime(data.datetime)
    data = data.drop(columns=['datetime'])

    mpf.plot(data, type='candle', volume=True, style=context.my_style, axisoff=True, tight_layout=True, savefig={'fname': fpath})


def init(context):
    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "START")
    config = configparser.ConfigParser()
    config.read('config.ini')

    context.PERIOD = config.getint('CANDLE', 'PERIOD')
    context.FREQUENCY = '1d'
    context.BAR_COUNT = 52*5 + context.PERIOD

    context.classifying = pd.read_csv("sample.csv", parse_dates=["order_day"], date_parser=lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
    # CONVERT dtype: datetime64[ns] to datetime.date
    context.classifying['order_day'] = context.classifying['order_day'].dt.date
    context.classifying = context.classifying.drop(["holding_days", "profit"], axis=1)

    test = shuffle(context.classifying).sample(frac=0.142857)
    context.classifying.loc[test.index, 'usage'] = 'test'
    context.classifying['usage'][context.classifying['usage'] != 'test'] = 'train'

    my_color = mpf.make_marketcolors(up='r', down='g', edge='inherit', wick='inherit', volume='inherit')
    context.my_style = mpf.make_mpf_style(marketcolors=my_color)

    context.class_map = {"loss": "0", "holding": "1", "profit": "2"}

    if not os.path.exists('train'):
        os.mkdir("train")
        for sub in context.class_map.values():
            os.mkdir(os.path.join("train", sub))
    if not os.path.exists('test'):
        os.mkdir("test")
        for sub in context.class_map.values():
            os.mkdir(os.path.join("test", sub))


def after_trading(context):
    day = context.now.date()
    stocks = all_instruments(type="CS")
    for order_book_id in stocks['order_book_id']:
        historys = history_bars(order_book_id, context.BAR_COUNT, context.FREQUENCY, fields=['datetime', 'open','close','high','low','volume'], include_now=True)

        if not historys.size: continue

        order_data = context.classifying[(context.classifying['order_book_id'] == order_book_id) &
                                     (context.classifying['order_day'] < day) &
                                     (context.classifying['classify'] != "")]

        for order_day in order_data['order_day'].sort_values(ascending=False):
            order_day64 = np.int64(order_day.strftime("%Y%m%d%H%M%S"))
            # 逐次缩小historys
            historys = historys[(historys['datetime'] <= order_day64)]
            # 数据不足
            if historys.size < context.PERIOD:
                break

            index = context.classifying[(context.classifying['order_book_id'] == order_book_id) &
                                      (context.classifying['order_day'] == order_day)].index[0]
            fpath = os.path.join(context.classifying['usage'][index],
                                 context.class_map[context.classifying['classify'][index]],
                                 '_'.join([order_day.strftime("%Y%m%d"), order_book_id[:6], '.jpg']))

            gen_kline(context, historys[-1-context.PERIOD:], fpath)

    if context.run_info.end_date == day:
        print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "END")

