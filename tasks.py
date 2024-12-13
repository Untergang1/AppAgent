class Difficulty:
    EASY = 3
    MEDIUM = 6
    HARD = 9


# task: Tuple[msg: str, max_rounds: int]
train_tasks = [
    ("take a photo", Difficulty.EASY),                  # EASY
    ("check new emails", Difficulty.EASY),
    ("check my plans in calender", Difficulty.EASY),
    ("turn on wifi", Difficulty.MEDIUM),                # MEDIUM
    ("turn on airplane mode", Difficulty.MEDIUM),
    ("turn on the dark mode", Difficulty.MEDIUM),
    ("search the weather tomorrow", Difficulty.MEDIUM),
    ("Send an empty email to to 330037594@qq.com.", Difficulty.HARD),   # HARD
]

eval_tasks = [

]

