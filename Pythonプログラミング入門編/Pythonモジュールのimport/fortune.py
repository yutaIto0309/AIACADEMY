#運勢を占うプログラム
from random import choice

def get_fortune():
    fortune_list = ["凶","吉","中吉","大吉"]
    return choice(fortune_list)
