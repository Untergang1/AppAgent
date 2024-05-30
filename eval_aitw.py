from scripts.parse_aitw import get_aitw, action_space
from PIL import Image, ImageDraw

max_episode = 50
dataset = get_aitw('general', max_episodes=max_episode)



