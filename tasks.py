class Difficulty:
    EASY = 3
    MEDIUM = 5
    HARD = 8


# task: Tuple[msg: str, max_rounds: int]

camera_tasks = {
    'train': [
        ("take a photo", Difficulty.EASY),
        ("take a picture with flash on", Difficulty.MEDIUM),
    ],
    'eval': [
        ("take a selfie", Difficulty.EASY),
        ("open camera and switch to front camera", Difficulty.EASY),
        ("take a group photo with timer", Difficulty.MEDIUM),
    ]
}

gmail_tasks = {
    'train': [
        ("check new emails", Difficulty.EASY),
        ("check spam folder", Difficulty.MEDIUM),
        ("open Gmail and compose a new email", Difficulty.MEDIUM),
        ("reply to an email", Difficulty.MEDIUM),
    ],
    'eval': [
        ("read the latest email", Difficulty.EASY),
        ("check my draft", Difficulty.MEDIUM),
        ("check my Outbox", Difficulty.MEDIUM),
        ("attach a file to an email", Difficulty.MEDIUM),
        ("flag an email as important", Difficulty.MEDIUM),
        ("send an email with a subject and attachments", Difficulty.HARD),
        ("send an email with multiple recipients", Difficulty.HARD),
    ]
}


calender_tasks = {
    'train': [
        ("check my plans in calendar", Difficulty.EASY),
        ("add an event to my calendar", Difficulty.MEDIUM),
        ("edit an event", Difficulty.MEDIUM),
        ("view events for the next week", Difficulty.MEDIUM),
    ],
    'eval': [
        ("add an all-day event", Difficulty.EASY),
        ("delete an event", Difficulty.MEDIUM),
        ("view calendar in week view", Difficulty.MEDIUM),
        ("set a task for an event", Difficulty.MEDIUM),
        ("create an event with repeating schedule", Difficulty.HARD),
        ("schedule a meeting at tomorrow", Difficulty.HARD),
    ]
}


settings_tasks = {
    'train': [
        ("turn on wifi", Difficulty.MEDIUM),
        ("adjust the brightness level", Difficulty.MEDIUM),
        ("enable location services", Difficulty.MEDIUM),
        ("change the wallpaper", Difficulty.HARD),
        ("change ringtone and vibration settings", Difficulty.HARD),
    ],
    'eval': [
        ("turn off bluetooth", Difficulty.MEDIUM),
        ("turn on airplane mode", Difficulty.MEDIUM),
        ("turn off mobile data", Difficulty.MEDIUM),
        ("turn on the dark mode", Difficulty.MEDIUM),
        ("set a do not disturb schedule", Difficulty.MEDIUM),
        ("turn off notifications for gmail app", Difficulty.HARD),
        ("set up a new Wi-Fi network", Difficulty.HARD),
    ]
}

browser_tasks = {
    'train': [
        ("search for weather tomorrow", Difficulty.EASY),
        ("open a specific website", Difficulty.MEDIUM),
        ("bookmark a website", Difficulty.MEDIUM),
    ],
    'eval': [
        ("open the browser", Difficulty.EASY),
        ("search for a recipe", Difficulty.EASY),
        ("clear browser history", Difficulty.MEDIUM),
        ("check the latest sports scores", Difficulty.MEDIUM),
        ("open incognito mode and search", Difficulty.MEDIUM),
    ]
}


all_app_tasks = {
    'camera': camera_tasks,
    'gmail': gmail_tasks,
    'calender': calender_tasks,
    'settings': settings_tasks,
    'browser': browser_tasks,
}

train_tasks = [task for app in all_app_tasks.values() for task in app['train']]     # 18 tasks

eval_tasks = [task for app in all_app_tasks.values() for task in app['eval']]       # 28 tasks

test_tasks = [
    ("turn on wifi", 4),
    ("add an event to my calendar", 5),
    ("open Gmail and send a new email to test@example.com", 4),
]