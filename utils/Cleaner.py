import os
import re
import datetime as dt

from env.running_env import milestone_base, image_base, exp_base, log_base
from utils.objectIO import fetch_file_name, dir_files, remove_files


class FileCleaner:
    ERROR_MESS1 = "Do not search date to satisfy the regular(FileCleaner.pattern)."
    ERROR_MESS2 = "Do not search date to satisfy the regular(FileCleaner.data_pattern)."

    def __init__(self, remain_days: int, year: int = 2022):
        self.curt_year = year
        self.remain = remain_days
        self.format_time = "%Y.%m.%d"
        self.pattern = re.compile(r"---([\d\.]+)\.[\w]+$|^\d+\.([\d\.]+)_", re.MULTILINE)
        self.data_pattern = re.compile(r"[\d\.]+", re.MULTILINE)

    def day_consumed(self, date: str) -> int:
        curt_date_str = dt.datetime.now().strftime(self.format_time)
        curt_date = dt.datetime.strptime(curt_date_str, self.format_time).date()
        date = dt.datetime.strptime(date, self.format_time).date()
        days = (curt_date - date).days
        return days

    def fetch_date(self, file_name: str) -> str:
        date = None
        match = self.pattern.search(file_name)
        assert match, self.ERROR_MESS1
        for group in match.groups():
            if group:
                if self.data_pattern.match(group):
                    date = group
        assert date, self.ERROR_MESS2
        return f"{self.curt_year}.{date}"

    def find_files(self) -> list:
        all_files = []
        files_path = []
        files_base = [milestone_base, log_base, image_base, exp_base]
        for base in files_base:
            all_files.extend(dir_files(base))

        for f_path in all_files:
            file_name = fetch_file_name(f_path)
            if self.pattern.search(file_name):
                if self.day_consumed(self.fetch_date(file_name)) > self.remain:
                    files_path.append(f_path)
        return files_path

    def clear_files(self):
        to_del = self.find_files()
        remove_files(to_del)
