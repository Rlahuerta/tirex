import datetime
from typing import Tuple, List


def create_time_index(delta_time, ini_time, size: int) -> List[datetime.datetime]:

    list_output_datetime_idx = []
    dt_current = ini_time
    for i in range(size):
        dt_current += delta_time
        list_output_datetime_idx.append(dt_current)

    return list_output_datetime_idx
