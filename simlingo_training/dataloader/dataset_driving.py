"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import os
import ujson
import json
import numpy as np
import random
import cv2
import re
import gzip

import torch
from simlingo_training.utils.custom_types import DatasetOutput
from simlingo_training.dataloader.dataset_base import BaseDataset


VIZ_DATA = False

class Data_Driving(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        super().__init__(dreamer=False, **cfg)

    @staticmethod
    def _distance_to_future_path(pos_xy, waypoints):
        if waypoints is None or len(waypoints) == 0:
            return float("inf")
        future_path = np.asarray(waypoints, dtype=np.float32)
        future_path = future_path[: min(len(future_path), 12)]
        return float(np.linalg.norm(future_path - pos_xy, axis=-1).min())

    @staticmethod
    def _future_path_turns(waypoints):
        if waypoints is None or len(waypoints) < 4:
            return False
        future_path = np.asarray(waypoints, dtype=np.float32)
        future_path = future_path[: min(len(future_path), 12)]
        lateral = future_path[:, 1]
        lateral_span = float(lateral.max() - lateral.min())
        final_lateral = float(abs(lateral[-1]))
        return lateral_span > 2.0 or final_lateral > 2.0

    @staticmethod
    def _expert_yields_from_waypoints(waypoints):
        """Approximate whether the expert slows/yields in the near future."""
        if waypoints is None or len(waypoints) < 4:
            return False

        displacements = np.linalg.norm(
            np.asarray(waypoints[1:], dtype=np.float32)
            - np.asarray(waypoints[:-1], dtype=np.float32),
            axis=-1,
        )
        if len(displacements) < 3:
            return False

        early_speed = float(np.mean(displacements[: min(3, len(displacements))]))
        late_speed = float(np.mean(displacements[-min(3, len(displacements)):]))
        min_future_speed = float(displacements[min(1, len(displacements) - 1):].min())

        will_stop = min_future_speed < 0.12
        clearly_slows = early_speed > 0.20 and late_speed < 0.65 * early_speed
        return will_stop or clearly_slows

    @staticmethod
    def _commentary_explains_rule_or_following_yield(commentary):
        """Return true when the commentary should not become a dynamic-yield label.

        The temporal interaction label should focus on dynamic actor conflicts,
        not on ordinary red-light, stop-sign, front-car-following,
        construction, or explicit go/accelerate commands.
        """
        if not commentary:
            return False

        text = commentary.lower()
        rule_yield = any(
            phrase in text
            for phrase in (
                'red traffic light',
                'traffic light is red',
                'because of a traffic light',
                'due to a traffic light',
                'stop sign',
            )
        )
        following_yield = (
            'stay behind' in text
            or 'to stay behind' in text
            or ('follow the' in text and 'to the front' in text)
            or ('behind the' in text and 'to the front' in text)
        )
        construction_context = 'construction' in text
        explicit_go = any(
            phrase in text
            for phrase in (
                'accelerate',
                'proceed',
                'drive through',
                'go through',
                'drive with the target speed',
                'reach the speed limit',
            )
        )
        return rule_yield or following_yield or construction_context or explicit_go

    def _get_actor_motion_labels(self, box_path, waypoints=None, commentary=None):
        """Build compact interaction labels from current-frame boxes.

        Labels:
        - moving front/path actor close to the ego future path
        - moving lateral/cross-traffic actor close to the ego future path
        - stopped actor blocking the ego future path
        - expert yield/slowdown for a moving actor conflict
        """
        try:
            with gzip.open(box_path, 'rt') as f:
                boxes = ujson.load(f)
        except (FileNotFoundError, ujson.JSONDecodeError):
            return None, ''

        labels = np.zeros(4, dtype=np.float32)
        closest_path_moving = None
        closest_side_moving = None
        closest_path_stopped = None

        for box in boxes:
            if box.get('class') not in ('car', 'walker'):
                continue
            pos = box.get('position', [0, 0, 0])
            pos_xy = np.asarray(pos[:2], dtype=np.float32)
            dist = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
            if dist > 25.0 or pos[0] < -5.0:
                continue

            path_distance = self._distance_to_future_path(pos_xy, waypoints)
            near_path = path_distance < 4.5 or (pos[0] > 0.0 and abs(pos[1]) < 4.0)
            side = abs(pos[1]) >= 3.0
            moving = box.get('speed', 0.0) > 0.5
            obj_type = 'vehicle' if box.get('class') == 'car' else 'pedestrian'

            if moving and near_path:
                item = (path_distance, dist, obj_type, pos[0], pos[1])
                if side:
                    labels[1] = 1.0
                    if closest_side_moving is None or item[:2] < closest_side_moving[:2]:
                        closest_side_moving = item
                else:
                    labels[0] = 1.0
                    if closest_path_moving is None or item[:2] < closest_path_moving[:2]:
                        closest_path_moving = item
            elif not moving and near_path:
                labels[2] = 1.0
                item = (path_distance, dist, obj_type, pos[0], pos[1])
                if closest_path_stopped is None or item[:2] < closest_path_stopped[:2]:
                    closest_path_stopped = item

        expert_yields = self._expert_yields_from_waypoints(waypoints)
        has_commentary_filter = bool(commentary and commentary.strip())
        rule_or_following_yield = self._commentary_explains_rule_or_following_yield(commentary)
        dynamic_side_actor = closest_side_moving is not None
        dynamic_turn_path_actor = False
        if closest_path_moving is not None and self._future_path_turns(waypoints):
            path_distance, dist, _, _, lateral = closest_path_moving
            # Capture oncoming/turning conflicts where the actor is not
            # lateral enough to be a "side" actor but lies on the curved ego
            # path. The lateral offset avoids relabeling ordinary same-lane
            # following as temporal yield supervision.
            dynamic_turn_path_actor = (
                path_distance < 2.5
                and dist < 25.0
                and abs(lateral) > 1.0
            )
        dynamic_conflict_actor = dynamic_side_actor or dynamic_turn_path_actor
        if dynamic_conflict_actor and expert_yields and has_commentary_filter and not rule_or_following_yield:
            labels[3] = 1.0

        parts = []
        if labels[3] > 0.0:
            parts.append('A moving actor is close to the ego path, so the ego should yield until the path is clear.')
        elif closest_side_moving is not None:
            _, _, obj_type, _, lateral = closest_side_moving
            side_name = 'left' if lateral < 0 else 'right'
            parts.append(f'A moving {obj_type} from the {side_name} is close to the ego path.')
        elif closest_path_moving is not None:
            _, _, obj_type, _, _ = closest_path_moving
            parts.append(f'A moving {obj_type} is close to the ego path.')
        elif closest_path_stopped is not None:
            _, _, obj_type, _, _ = closest_path_stopped
            parts.append(f'A stopped {obj_type} is close to the ego path.')

        return labels, ' '.join(parts)

    @staticmethod
    def _remove_conflicting_junction_claims(commentary, actor_motion_labels):
        """Remove stale junction-clear/stopped claims contradicted by boxes.

        Some original commentary templates say that other vehicles are stopped
        or the junction is clear. If current boxes show a moving actor near the
        ego path, keeping those sentences creates contradictory supervision.
        """
        if actor_motion_labels is None:
            return commentary

        moving_actor_near_path = actor_motion_labels[0] > 0.0 or actor_motion_labels[1] > 0.0
        if not moving_actor_near_path:
            return commentary

        stale_junction_claims = (
            'other vehicles are stopped at the junction',
            'vehicles are stopped at the junction',
            'traffic is stopped at the junction',
            'vehicle at the junction is moving away',
            'vehicle in the junction is moving away',
            'vehicles in the junction are moving away',
            'junction is clear',
            'clear junction',
        )
        proceed_words = (
            'accelerate',
            'proceed',
            'drive through',
            'go through',
        )
        sentences = re.split(r'(?<=[.!?])\s+', commentary.strip())
        kept = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            says_to_enter = any(word in sentence_lower for word in proceed_words)
            has_stale_claim = any(claim in sentence_lower for claim in stale_junction_claims)
            green_light_junction_go = (
                says_to_enter
                and 'traffic light is green' in sentence_lower
                and 'junction' in sentence_lower
            )
            # Drop the whole stale sentence. This is especially important for
            # templates such as "Accelerate because the traffic light is green
            # and the junction is clear", where keeping only the green-light
            # part would still teach the wrong behavior.
            if has_stale_claim or green_light_junction_go:
                continue
            kept.append(sentence)

        if not kept:
            return 'Follow the route.'
        return ' '.join(kept)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]

        ######################################################
        ######## load current and future measurements ########
        ######################################################
        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(
            measurements,
            sample_start
            )
        
        data['measurement_path'] = measurement_file_current

        # Determine whether the augmented camera or the normal camera is used.
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement['augmentation_rotation']
            aug_translation = current_measurement['augmentation_translation']
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0


        ######################################################
        ################## load waypoints ####################
        ######################################################
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)
       
        speed_rounded = round(current_measurement['speed'], 1)
        data['speed'] = current_measurement['speed']

        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)

        current_box_path = str(
            self.boxes[index][self.hist_len - 1],
            encoding='utf-8',
        )
        actor_motion_labels = None
        actor_motion_context = ''
        if getattr(self, 'use_motion_descriptions', False):
            actor_motion_labels, actor_motion_context = self._get_actor_motion_labels(
                current_box_path,
                data['waypoints_org'],
            )

        target_point = np.array(current_measurement['target_point'])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        next_target_point = np.array(current_measurement['target_point_next'])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)

        ######################################################
        ################## get commentary & qa ##################
        ######################################################
        commentary_exists = False
        commentary = ''
        if self.use_commentary:
            commentary_file_path = measurement_file_current.replace('measurements', 'commentary').replace('data/', 'commentary/') # TODO: move to config
            # do not use evaluation routes!!!
            if 'validation_' in commentary_file_path:
                commentary_exists = False
            else:
                try:
                    with gzip.open(commentary_file_path, 'rt') as f:
                        commentary_file = ujson.load(f)
                        commentary_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    commentary_exists = False
                    commentary_file = None

                if commentary_file is not None:

                    commentary = commentary_file['commentary']
                    # we only augment in 60% of the cases and use the default commentary in 40% of the cases
                    # augmentation is used to increase generalization to a broader set of sentences
                    # but we do not want to overfit to the augmented sentences
                    if self.commentary_augmentation and random.random() < 0.6:
                        if commentary_file['commentary_template'] in self.templates_commentary:
                            commentary = random.choice(self.templates_commentary[commentary_file['commentary_template']])
                            for key, value in commentary_file['placeholder'].items():
                                if key in commentary:
                                    commentary = commentary.replace(key, value)
                            # regex check if <OBJECT> or <LOCATION> or any other <> is still in commentary, if so use default commentary_file['commentary']
                            if re.search(r'<.*?>', commentary):
                                print(f"WARNING: {commentary} contains placeholders that are not replaced. Using default commentary.")
                                commentary = commentary_file['commentary']

                    commentary = commentary.replace('..', '.')
                    commentary = commentary.replace('in in', 'in')

                    # Append motion descriptions from bounding-box data so the
                    # language loss teaches the model to describe vehicle motion
                    # states (moving / stopped) – addresses Cause 3.
                    if getattr(self, 'use_motion_descriptions', False):
                        commentary = self._remove_conflicting_junction_claims(
                            commentary,
                            actor_motion_labels,
                        )
                        actor_motion_labels, actor_motion_context = self._get_actor_motion_labels(
                            current_box_path,
                            data['waypoints_org'],
                            commentary,
                        )
                        motion_ctx = actor_motion_context
                        if motion_ctx:
                            commentary = commentary.rstrip('.') + '. ' + motion_ctx
        
        qa_exists = False
        if self.use_qa:
            qa_path = measurement_file_current.replace('measurements', 'vqa').replace('data/', 'drivelm/')
            if 'validation_' in qa_path:
                qa_exists = False
            else:
                try:
                    with gzip.open(qa_path, 'rt') as f:
                        qa = ujson.load(f)
                    qa_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    qa_exists = False
                    qa = None

            if qa_exists:
                qas = qa['QA']
                qas = [values for values in qas.values()] # list of lists
                qas = [item for sublist in qas for item in sublist] # flatten list
                while True:
                    qa_chosen = random.choice(qas)
                    qa_question = qa_chosen['Q']
                    qa_answer = qa_chosen['A']
                    
                    # TODO: make this nicer!!!
                    if 'There are no pedestrians.' in qa_answer or \
                            'There is no traffic light' in qa_answer or \
                            'There are no pedestrians.' in qa_answer or \
                            'No, the ego vehicle is not affected by a stop sign.' in qa_answer or \
                            'No, the ego vehicle is not affected by a junction.' in qa_answer or \
                            'There is no traffic light affecting the ego vehicle.' in qa_answer or \
                            'There is no stop sign affecting the ego vehicle.' in qa_answer or \
                            'There is no junction affecting the ego vehicle.' in qa_answer or \
                            'It is not possible to tell' in qa_answer or \
                            'There is no reason for the ego vehicle to brake.' in qa_answer:
                        # only keep in 20% of the cases
                        if random.random() < 0.2:
                            break
                    else:
                        break
                
                # we only augment in 60% of the cases and use the default QA in 40% of the cases
                # augmentation is used to increase generalization to a broader set of sentences
                # but we do not want to overfit to the augmented sentences
                if self.qa_augmentation and random.random() < 0.6:
                    qa_question_org = qa_question
                    qa_answer_org = qa_answer
                    locations = [
                        'nearby to the front of the ego vehicle',
                        'nearby to the front right of the ego vehicle',
                        'nearby to the front left of the ego vehicle',
                        'nearby on the left side of the ego vehicle',
                        'far to the front left of the ego vehicle',
                        'far to the front right of the ego vehicle',
                        'far to the front of the ego vehicle',
                        'far to the left side of the ego vehicle',
                        'far to the right side of the ego vehicle',
                        'to the front of the ego vehicle',
                        'to the front right of the ego vehicle',
                        'to the front left of the ego vehicle',
                        'on the left side of the ego vehicle',
                        'on the right side of the ego vehicle',
                    ]
                    objects = [value['Visual_description'] for key, value in qa['key_object_infos'].items()]
                    q_objects = []
                    a_objects = []
                    for object_type in objects:
                        if object_type in qa_question:
                            qa_question = qa_question.replace(object_type, '<OBJECT>')
                            q_objects.append(object_type)
                        if object_type in qa_answer:
                            qa_answer = qa_answer.replace(object_type, '<OBJECT>')
                            a_objects.append(object_type)
                    
                    q_location = ''
                    a_location = ''
                    for location in locations:
                        if location in qa_question:
                            qa_question = qa_question.replace(location, '<LOCATION>')
                            q_location = location
                        if location in qa_answer:
                            qa_answer = qa_answer.replace(location, '<LOCATION>')
                            a_location = location
                        
                    q_distance = re.search(r'in (\d+) m', qa_question)
                    qa_question = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_question)
                    a_distance = re.search(r'in (\d+) m', qa_answer)
                    qa_answer = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_answer)
                    if len(q_objects)==0:
                        q_objects = ['']
                    if len(a_objects)==0:
                        a_objects = ['']
                    
                    # in 40% of the cases we do not augment the question
                    if len(q_objects) > 1 or len(a_objects) > 1 or random.random() < 0.4: 
                        qa_question = qa_question_org
                        qa_answer = qa_answer_org
                    else:
                        if qa_question in self.q_augment:
                            qa_question = random.choice(self.q_augment[qa_question]).replace('<OBJECT>', q_objects[0]).replace('<LOCATION>', q_location)
                            if q_distance:
                                qa_question = qa_question.replace('<DISTANCE>', q_distance.group(1))
                        else:
                            print(f"WARNING: {qa_question} not in q_augment. Using default question.")
                            qa_question = qa_question_org
                        if qa_answer in self.a_augment:
                            qa_answer = random.choice(self.a_augment[qa_answer]).replace('<OBJECT>', a_objects[0]).replace('<LOCATION>', a_location)
                            if a_distance:
                                qa_answer = qa_answer.replace('<DISTANCE>', a_distance.group(1))
                        else:
                            print(f"WARNING: {qa_answer} not in a_augment. Using default answer.")
                            qa_answer = qa_answer_org

        ######################################################
        ######## load navigational_conditioning ########
        ######################################################
        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)
            
        answer = ''

        prompt_random = random.random()
        motion_prompt = 'Consider nearby traffic motion and whether the ego path is clear. ' if getattr(self, 'use_motion_prompt', False) else ''
        
        if self.use_commentary and commentary_exists and prompt_random < self.prompt_probabilities['commentary']:
            if random.random() < 0.2: # 20% of the time we give commentary as prompt
                if random.random() < 0.5:
                    prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {commentary} {motion_prompt}Predict the waypoints."
                else:
                    prompt = f"Current speed: {speed_rounded} m/s. Command: {commentary} {motion_prompt}Predict the waypoints."
                answer = f"Waypoints:"
            else:
                # 80% of the time we want to predict commentary
                prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {motion_prompt}What should the ego do next?"
                answer = f"{commentary} Waypoints:"
            self.num_sampled_per_type['commentary'] += 1
            
        elif self.use_qa and qa_exists and prompt_random < (self.prompt_probabilities['qa'] + self.prompt_probabilities['commentary']):
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Q: {qa_question}"
            answer = f"A: {qa_answer}"
            self.num_sampled_per_type['qa'] += 1
            
        else:
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {motion_prompt}Predict the waypoints."
            answer = f"Waypoints:"
            self.num_sampled_per_type['driving'] += 1

        # recalculate the probabilties after warmup (when more than 1000 samples have been sampled)
        # we do this in case we dont have qa or commentary for every sample otherwise it would lead to undersampling one of those
        if sum(self.num_sampled_per_type.values()) > 10000 and sum(self.num_sampled_per_type.values()) % 10000 == 0:
            self.prompt_probabilities = {key: 1/value for key, value in self.num_sampled_per_type.items()}
            self.prompt_probabilities = {key: value/sum(self.prompt_probabilities.values()) for key, value in self.prompt_probabilities.items()}
            print(f"Prompt probabilities: {self.prompt_probabilities}")
            print(f"Number of samples per type: {self.num_sampled_per_type}")
            
        answer = answer.replace('..', '.')
        prompt = prompt.replace('..', '.')

        ######################################################
        ######## load current and past images ########
        ######################################################
        data = self.load_images(data, images, augment_sample=augment_sample)
        

        conversation_answer = [
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            },
        ]
        conversation_all = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            }
        ]
        
        images = [data['rgb']]

        # Compute motion label for auxiliary temporal loss (Cause 1).
        motion_eval_infos = None
        if getattr(self, 'use_motion_descriptions', False) and actor_motion_labels is not None:
            motion_eval_infos = {
                'actor_motion_labels': actor_motion_labels,
                'actor_motion_mask': 1.0,
                'interaction_sample_weight_mask': float(actor_motion_labels[3] > 0.0),
            }

        data_new = DatasetOutput(
            conversation = conversation_all,
            answer = conversation_answer,
            image_ff = data['rgb'],
            image_ff_org_size=data['rgb_org_size'],
            waypoints = data["waypoints"],
            waypoints_1d = data["waypoints_1d"],
            path = data['route_adjusted'],
            target_points = data['target_points'],
            speed = data['speed'],
            placeholder_values = placeholder_values,
            measurement_path = data['measurement_path'],
            dataset = 'driving',
            eval_infos = motion_eval_infos,
        )
        
        if VIZ_DATA:
            # front image with path and waypoints and commentary
            self.visualise_cameras(data_new, commentary, data['route_adjusted'], data['waypoints'], options=None, prompt=prompt, answer=answer, name="img")
        return data_new


if __name__ == "__main__":
    from hydra import compose, initialize
    from simlingo_training.config import TrainConfig
    
    # seed all
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    

    initialize(config_path="../config")
    cfg = compose(config_name="config")
    
    cfg.data_module.base_dataset.use_commentary = True
    cfg.data_module.base_dataset.use_qa = True
    cfg.data_module.base_dataset.img_shift_augmentation = False

    print('Test Dataset')
    dataset = Data_Driving(                        
                        split="train",
                        bucket_name='all',
                        **cfg.data_module,
                        **cfg.data_module.base_dataset,
    )

    for i in range(len(dataset)):
        # shuffle
        # i = np.random.randint(0, len(dataset))
        data = dataset[i]
        # print(data)
        # if i == 100:
        #     break
