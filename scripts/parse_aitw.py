import tensorflow as tf
from PIL import Image
from IPython.display import display

action_space = [
    "UNUSED_0",             # 0
    "UNUSED_1",             # 1
    "UNUSED_2",             # 2
    "TYPE",                 # 3
    "DUAL_POINT",           # 4
    "PRESS_BACK",           # 5
    "PRESS_HOME",           # 6
    "PRESS_ENTER",          # 7
    "UNUSED_8",             # 8
    "UNUSED_9",             # 9
    "STATUS_TASK_COMPLETE", # 10
    "STATUS_TASK_IMPOSSIBLE" # 11
]

def get_aitw(dataset_name, max_episodes=None):
    dataset_directories = {
        'general': './datasets/AITW/general/*',
        'google_apps': './datasets/AITW/google_apps/*',
        'install': './datasets/AITW/install/*',
        'single': './datasets/AITW/single/*',
        'web_shopping': './datasets/AITW/web_shopping/*',
    }

    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

    dataset = []
    current_episode = []
    current_episode_id = None
    num_episodes = 1

    for d in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(d)

        android_api_level = example.features.feature['android_api_level'].int64_list.value[0]
        current_activity = example.features.feature['current_activity'].bytes_list.value[0]
        device_type = example.features.feature['device_type'].bytes_list.value[0]
        episode_id = example.features.feature['episode_id'].bytes_list.value[0]
        episode_length = example.features.feature['episode_length'].int64_list.value[0]
        goal_info = example.features.feature['goal_info'].bytes_list.value[0]
        step_id = example.features.feature['step_id'].int64_list.value[0]

        image_height = example.features.feature['image/height'].int64_list.value[0]
        image_width = example.features.feature['image/width'].int64_list.value[0]
        image_channels = example.features.feature['image/channels'].int64_list.value[0]
        image = tf.io.decode_raw(
            example.features.feature['image/encoded'].bytes_list.value[0],
            out_type=tf.uint8,
        )
        height = tf.cast(image_height, tf.int32)
        width = tf.cast(image_width, tf.int32)
        n_channels = tf.cast(image_channels, tf.int32)
        image = tf.reshape(image, (height, width, n_channels))
        image_np = image.numpy()

        ui_annotations_positions = example.features.feature['image/ui_annotations_positions'].float_list.value
        ui_annotations_text = example.features.feature['image/ui_annotations_text'].bytes_list.value
        ui_annotations_ui_types = example.features.feature['image/ui_annotations_ui_types'].bytes_list.value
        results_action_type = example.features.feature['results/action_type'].int64_list.value[0]
        results_type_action = example.features.feature['results/type_action'].bytes_list.value[0]
        results_yx_touch = example.features.feature['results/yx_touch'].float_list.value
        results_yx_lift = example.features.feature['results/yx_lift'].float_list.value

        step = {
            'image': image_np,
            'step_id': step_id,
            'android_api_level': android_api_level,
            'current_activity': current_activity,
            'device_type': device_type,
            'episode_id': episode_id,
            'episode_length': episode_length,
            'goal_info': goal_info,
            'ui_annotations_positions': ui_annotations_positions,
            'ui_annotations_text': ui_annotations_text,
            'ui_annotations_ui_types': ui_annotations_ui_types,
            'results_action_type': results_action_type,
            'results_type_action': results_type_action,
            'results_yx_touch': results_yx_touch,
            'results_yx_lift': results_yx_lift,
        }

        if current_episode_id is None:
            current_episode_id = episode_id

        if episode_id != current_episode_id:
            num_episodes += 1
            dataset.append(current_episode)
            current_episode = []
            current_episode_id = episode_id

        if max_episodes and num_episodes > max_episodes:
            break

        current_episode.append(step)

    if current_episode:
        dataset.append(current_episode)

    return dataset

if __name__ == '__main__':
    dataset = get_aitw('general', max_episodes=50)
    episode = dataset[0]
    for example in episode:
        for key, value in example.items():
            if key == 'image':
                display(value)
            elif key == 'results_action_type':
                print(key)
                print(action_space[value])
                print()
            else:
                print(key)
                print(value)
                print()