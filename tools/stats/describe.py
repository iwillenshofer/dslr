from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from .stdev import stdev, mean
import os
from .percentile import percentile
from .fill_na import remove_na


class Describe():
    """Class to create a description of a DataFrame"""
    TITLE_ROW_WIDTH = 5

    def __init__(self, ds: DataFrame):
        """initializes the class"""
        self.describe(ds)

    def __repr__(self):
        return "Describe()"

    def __str__(self):
        table = []
        table.append([''] + self.keys)
        for i in self.values.keys():
            table.append([i] + ['%.6f' % z for z in self.values[i]])
        column_widths = [self.TITLE_ROW_WIDTH] \
            + [len(i) + 2 for i in self.keys]
        for z in table:
            for idx, _ in enumerate(z):
                size = len(z[idx]) + (2 if idx else 0)
                if size > column_widths[idx]:
                    column_widths[idx] = size
        reduced = self.reduce_table(table, column_widths)
        for z in table:
            for i, _ in enumerate(z):
                w = column_widths[i]
                z[i] = z[i][:w].rjust(w) if i > 0 else z[i][:w].ljust(w)
        if reduced:
            table.append("")
            table.append(
                f"[{len(self.values)} rows x {len(self.keys)} columns]")
        return '\n'.join([''.join(i) for i in table])

    def reduce_table(self, table, widths):
        """depending on table size and screen width, table will be reduced"""
        screen_width = 80
        try:
            screen_width, _ = os.get_terminal_size()
        except Exception:
            pass
        total_width = sum(widths)
        if total_width < screen_width:
            return False
        left, right = 0, 0
        center = int(len(widths)) // 2
        while (sum(widths) + 5 > screen_width and center - left > 1):
            widths[center - left] = 0
            if not right and len(widths) % 2:
                right += 1
            widths[center + right] = 0
            left += 1
            right += 1
        widths[center] = 5
        for z in table:
            z[center] = "...".rjust(5)
        return True

    def info(self, data: list) -> dict:
        """
        sorts data and calculates min, max, percentiles...
        returns a dictionary with its description:
        keys = ['count', 'mean', 'std', 'min', 'max', '25%', '50%', '75%']
        """
        d = remove_na(data)
        d = sorted(d)
        dic = {}
        if not data:
            raise AssertionError("List must not be empty")
        dic["Count"] = len(d)
        dic["Mean"] = mean(d)
        dic["Std"] = stdev(d)
        dic["Min"] = d[0] if len(d) > 1 else float('nan')
        dic["Max"] = d[-1] if len(d) > 1 else float('nan')
        dic["25%"], dic["50%"], dic["75%"] = percentile(d, [25, 50, 75])
        dic["Nan"] = len(data) - len(d)
        return (dic)

    def describe(self, ds: DataFrame):
        self.keys = []
        self.values = {
            "Count": [],
            "Mean": [],
            "Std": [],
            "Min": [],
            "25%": [],
            "50%": [],
            "75%": [],
            "Max": [],
            "Nan": []
        }
        for i in ds:
            if is_numeric_dtype(ds[i]):
                self.keys.append(i)
                row = list(ds[i])
                ret = self.info(row)
                for k in ret.keys():
                    self.values[k].append(ret[k])
