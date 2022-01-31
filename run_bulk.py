# -*- coding: utf-8 -*-

from rqalpha import run_file

import datetime as dt

start_year = 2021
end_year = 2021

for year in range(start_year, end_year+1):
  # 跳过元旦
  day = 28

  # 跳过周末
  while dt.datetime(year, 12, day).isoweekday() > 5:
    day -= 1

  config = {
    "base": {
      "start_date": f"{year}-12-{day}",
      "end_date": f"{year}-12-{day}"
    },
    "extra": {
      "log_level": "warning",
    },
    "mod": {
      "sys_analyser": {
        "enabled": False
      }
    },
  }

  strategy_file_path = "./bulk.py"

  run_file(strategy_file_path, config)
