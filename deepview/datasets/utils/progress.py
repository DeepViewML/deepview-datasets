# Copyright 2024 by Au-Zone Technologies.  All Rights Reserved.
#
#  DUAL-LICENSED UNDER AGPL-3.0 OR DEEPVIEW AI MIDDLEWARE COMMERCIAL LICENSE
#    CONTACT AU-ZONE TECHNOLOGIES <INFO@AU-ZONE.COM> FOR LICENSING DETAILS


from typing import Iterable, Any

class ProgressBar:
    RED = '\33[91m'
    GREEN = '\33[92m'
    ORANGE = '\33[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'
    OK = u'\u2713'

    def __init__(
        self,
        desc: str,
        steps: int,
        size: int = 30,
        color: str = None,
        fill: str = None,
        empty: str = None,
        suffix: str = None

    ) -> None:
        colors = {'red': ProgressBar.RED, 'green': ProgressBar.GREEN, 'blue': ProgressBar.BLUE, 'orange': ProgressBar.ORANGE}
        self.fill = '█' if fill is None else fill
        self.empty = '-' if  empty is None else empty
        self.size = size
        self.color = colors.get(color)
        self.description = desc
        self.__current__ = 0
        self.total = steps
        self.suffix = "" if suffix is None else suffix
        self.run()

    def run(self):
        percent = ("{0:." + str(2) + "f}").format(100 * (self.__current__ / float(self.total)))
        filled_length = int(self.size * self.__current__ // self.total)
        bar = self.fill * filled_length + self.empty * (self.size - filled_length)
        print(f'\r{self.description}{self.color}{bar} {ProgressBar.ENDC}{percent}% {self.suffix}', end = "\r")
        # Print New Line on Complete
        if self.__current__ == self.total: 
            print()


    def update(
        self, 
        suffix = ""
    ) -> None:
        self.suffix = suffix
        self.__current__ += 1
        self.run()
    
    def finish(self):
        self.__current__ = 0



class FillingSquaresBar(ProgressBar):
    def __init__(
            self, 
            desc: str, 
            steps: int, 
            size: int = 30, 
            color: str = None, 
            suffix: str = None
        ) -> None:
        super().__init__(desc, steps, size, color, '▣', '▢', suffix)

class FillingCirclesBar(ProgressBar):
    def __init__(
            self, 
            desc: str, 
            steps: Iterable, 
            size: int = 30, 
            color: str = None, 
            suffix: str = None
        ) -> None:
        super().__init__(desc, steps, size, color, '◉', '◯', suffix)

