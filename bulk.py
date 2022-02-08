from rqalpha.apis import *
from sklearn.utils import shuffle

import datetime as dt
import mplfinance as mpf
import numpy as np
import pandas as pd

import configparser
import pathlib


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
    context.BAR_COUNT = (dt.datetime.strptime(config.get("PICK", "END_DAY"), "%Y-%m-%d") - dt.datetime.strptime(config.get("PICK", "START_DAY"), "%Y-%m-%d")).days
    context.BAR_COUNT = int(context.BAR_COUNT/7*5)

    context.classifying = pd.read_csv("classifying.csv", parse_dates=["order_day"], date_parser=lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
    # CONVERT dtype: datetime64[ns] to datetime.date
    context.classifying['order_day'] = context.classifying['order_day'].dt.date

    # 保留需要的列
    context.classifying = context.classifying[['order_day', 'order_book_id', 'classify']]

    context.classifying = context.classifying.drop(context.classifying[context.classifying["classify"].isna()].index)

    # 获取子类别
    sub_dirs = set(context.classifying["classify"])

    for sub in sub_dirs:
        temp = context.classifying[context.classifying['classify'] == sub]
        max_num = config.getint('BULK', 'NUMBER')
        if temp.count() > max_num:
            temp = shuffle(temp).sample(temp.count()-max_num)
            context.classifying = context.classifying.drop(temp.index)

    # 1/7用于测试
    for sub in sub_dirs:
        test = shuffle(context.classifying[context.classifying['classify'] == sub]).sample(frac=0.142857)
        context.classifying.loc[test.index, 'usage'] = 'test'
    # 6/7用于训练
    train = context.classifying[context.classifying['usage'] != 'test']
    context.classifying.loc[train.index, 'usage'] = 'train'

    my_color = mpf.make_marketcolors(up='r', down='g', edge='inherit', wick='inherit', volume='inherit')
    context.my_style = mpf.make_mpf_style(marketcolors=my_color)

    if not pathlib.Path('train').exists():
        for sub in sub_dirs:
            pathlib.Path().joinpath("train", sub).mkdir(parents=True)
    if not pathlib.Path('test').exists():
        for sub in sub_dirs:
            pathlib.Path().joinpath("test", sub).mkdir(parents=True)


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
            fpath = pathlib.Path().joinpath(context.classifying['usage'][index],
                                            context.classifying['classify'][index],
                                            '.'.join(['_'.join([order_day.strftime("%Y%m%d"),
                                                                order_book_id[:6]]),
                                                      'jpg']))

            gen_kline(context, historys[-1-context.PERIOD:], str(fpath))

    if context.run_info.end_date == day:
        print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "END")

