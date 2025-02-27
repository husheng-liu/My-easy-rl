import os
from gym.wrappers.record_video import RecordVideo
from gym.wrappers.monitoring.video_recorder import VideoRecorder



# class CustomRecordVideo(RecordVideo):
#     def __init__(self, env, video_folder, record_trigger):
#         super().__init__(env, video_folder=video_folder)
#         self.record_trigger = record_trigger
#         self.current_episode = 0

#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         if self.record_trigger(self.current_episode):
#             self._start_video_recorder()
#         if self.video_recorder:
#             self.video_recorder.capture_frame()
#         return observation, reward, terminated, truncated, info

#     def reset(self, **kwargs):
#         observation, info = self.env.reset(**kwargs)
#         if self.record_trigger(self.current_episode):
#             self._start_video_recorder()
#         if self.video_recorder:
#             self.video_recorder.capture_frame()
#         return observation, info

#     def close(self):
#         if self.video_recorder:
#             self.video_recorder.close()
#         self.env.close()

#     def _start_video_recorder(self):
#         if not self.video_recorder:
#             base_path = os.path.join(self.video_folder, f"rl-video-{self.current_episode}")
#             self.video_recorder = VideoRecorder(
#                 env=self.env,
#                 base_path=base_path,
#                 metadata={'episode': self.current_episode},
#                 enabled=True
#             )

#     def increment_episode(self):
#         self.current_episode += 1
#         if self.video_recorder:
#             self.video_recorder.close()
#             self.video_recorder = None

from gym.wrappers.record_video import RecordVideo

class CustomRecordVideo(RecordVideo):
    def __init__(self, env, video_folder, record_trigger):
        super().__init__(env, video_folder)
        self.record_trigger = record_trigger
        self._is_trigger_on = False

    def reset(self, **kwargs):
        print(f"self.episode_trigger: {self.episode_id}")
        if self.record_trigger(self.episode_id):
            self._is_trigger_on = True
        else:
            self._is_trigger_on = False
        return super().reset(**kwargs)

    def step(self, action):
        if self._is_trigger_on:
            return super().step(action)
        else:
            return self.env.step(action)