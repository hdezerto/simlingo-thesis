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
    def _load_boxes(box_path):
        try:
            with gzip.open(box_path, 'rt') as f:
                return ujson.load(f)
        except (FileNotFoundError, OSError, ujson.JSONDecodeError):
            return None

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
    def _normalise_light_state(state):
        if state is None:
            return None
        if isinstance(state, (int, np.integer)):
            return {0: 'red', 1: 'yellow', 2: 'green'}.get(int(state))
        text = str(state).lower()
        if 'red' in text or text == '0':
            return 'red'
        if 'yellow' in text or text == '1':
            return 'yellow'
        if 'green' in text or text == '2':
            return 'green'
        return None

    @staticmethod
    def _truthy(value):
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return bool(value)

    @staticmethod
    def _safe_float(value, default=None):
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_relevant_actor(box):
        return box.get('class') in ('car', 'walker')

    @staticmethod
    def _item_is_closer(candidate, current):
        return current is None or candidate[:2] < current[:2]

    @staticmethod
    def _ids_match(left, right):
        if left in (None, -1) or right in (None, -1):
            return False
        try:
            return int(left) == int(right)
        except (TypeError, ValueError):
            return False

    def _traffic_light_state_from_scene(self, boxes, current_measurement=None):
        candidates = []
        for box in boxes:
            cls = box.get('class')
            if cls not in ('traffic_light', 'traffic_light_vqa'):
                continue
            affects_ego = self._truthy(box.get('affects_ego'))
            if not affects_ego:
                continue
            distance = self._safe_float(box.get('distance'), float('inf'))
            if distance > 55.0:
                continue
            state = self._normalise_light_state(
                box.get('state', box.get('state_str'))
            )
            if state is not None:
                candidates.append((distance, state))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]

        for box in boxes:
            if box.get('class') == 'ego_info':
                state = self._normalise_light_state(box.get('traffic_light_state'))
                if state is not None:
                    return state

        if current_measurement and self._truthy(current_measurement.get('light_hazard')):
            return 'red'
        return None

    def _has_affecting_stop_sign(self, boxes, current_measurement=None):
        if current_measurement and self._truthy(current_measurement.get('stop_sign_hazard')):
            return True
        for box in boxes:
            if box.get('class') not in ('stop_sign', 'stop_sign_vqa'):
                continue
            if self._truthy(box.get('affects_ego')) and self._safe_float(box.get('distance'), float('inf')) < 35.0:
                return True
        return False

    @staticmethod
    def _box_in_or_near_junction(box):
        if Data_Driving._truthy(box.get('is_in_junction')):
            return True
        distance_to_junction = Data_Driving._safe_float(box.get('distance_to_junction'))
        return distance_to_junction is not None and distance_to_junction < 15.0

    def _ego_near_junction(self, boxes, current_measurement=None, waypoints=None):
        if current_measurement and self._truthy(current_measurement.get('junction')):
            return True
        for box in boxes:
            if box.get('class') != 'ego_info':
                continue
            if self._truthy(box.get('is_in_junction')) or self._truthy(box.get('is_intersection')):
                return True
            distance_to_junction = self._safe_float(box.get('distance_to_junction'))
            if distance_to_junction is not None and distance_to_junction < 20.0:
                return True
        return self._future_path_turns(waypoints)

    @staticmethod
    def _is_same_lane_lead_vehicle(box, path_distance):
        if box.get('class') != 'car':
            return False
        pos = box.get('position', [0, 0, 0])
        if len(pos) < 2 or pos[0] <= 0.0:
            return False
        distance = Data_Driving._safe_float(box.get('distance'))
        if distance is None:
            distance = float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
        if distance > 35.0:
            return False

        same_road = box.get('same_road_as_ego')
        if same_road is not None and not Data_Driving._truthy(same_road):
            return False
        same_direction = box.get('same_direction_as_ego')
        if same_direction is not None and not Data_Driving._truthy(same_direction):
            return False
        lane_relative = Data_Driving._safe_float(box.get('lane_relative_to_ego'))
        if lane_relative is not None:
            return abs(lane_relative) < 0.5
        return abs(pos[1]) < 2.5 or path_distance < 3.0

    def _build_interaction_scene_facts(self, boxes, waypoints=None, current_measurement=None):
        """Compute scene facts once, then derive labels and text from them."""
        labels = np.zeros(4, dtype=np.float32)
        closest_path_moving = None
        closest_side_moving = None
        closest_path_stopped = None
        closest_lead_vehicle = None
        moving_junction_actor = False

        future_path_turns = self._future_path_turns(waypoints)
        expert_yields = self._expert_yields_from_waypoints(waypoints)
        ego_near_junction = self._ego_near_junction(boxes, current_measurement, waypoints)
        affecting_light_state = self._traffic_light_state_from_scene(boxes, current_measurement)
        light_hazard = bool(
            current_measurement
            and self._truthy(current_measurement.get('light_hazard'))
        )
        verified_traffic_stop = affecting_light_state in ('red', 'yellow') or light_hazard
        verified_green_light = affecting_light_state == 'green' and not light_hazard
        verified_stop_sign = self._has_affecting_stop_sign(boxes, current_measurement)
        speed_reduced_type = str(
            current_measurement.get('speed_reduced_by_obj_type', '')
            if current_measurement else ''
        ).lower()
        verified_construction = 'trafficwarning' in speed_reduced_type or 'construction' in speed_reduced_type
        speed_reduced_by_obj_id = current_measurement.get('speed_reduced_by_obj_id') if current_measurement else None
        vehicle_affecting_id = current_measurement.get('vehicle_affecting_id') if current_measurement else None

        for box in boxes:
            if not self._is_relevant_actor(box):
                continue
            pos = box.get('position', [0, 0, 0])
            if len(pos) < 2:
                continue
            pos_xy = np.asarray(pos[:2], dtype=np.float32)
            dist = self._safe_float(box.get('distance'))
            if dist is None:
                dist = float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
            if dist > 35.0 or pos[0] < -8.0:
                continue

            path_distance = self._distance_to_future_path(pos_xy, waypoints)
            near_path = path_distance < 4.5 or (pos[0] > 0.0 and abs(pos[1]) < 4.0)
            side = abs(pos[1]) >= 3.0
            moving = self._safe_float(box.get('speed'), 0.0) > 0.5
            obj_type = 'vehicle' if box.get('class') == 'car' else 'pedestrian'
            item = (path_distance, dist, obj_type, pos[0], pos[1], box)

            if moving and (near_path or (ego_near_junction and self._box_in_or_near_junction(box))):
                moving_junction_actor = True

            if dist > 25.0 or pos[0] < -5.0:
                continue

            if self._is_same_lane_lead_vehicle(box, path_distance):
                lead_item = (dist, path_distance, obj_type, pos[0], pos[1], box)
                if self._item_is_closer(lead_item, closest_lead_vehicle):
                    closest_lead_vehicle = lead_item
            if moving and near_path:
                if side:
                    labels[1] = 1.0
                    if self._item_is_closer(item, closest_side_moving):
                        closest_side_moving = item
                else:
                    labels[0] = 1.0
                    if self._item_is_closer(item, closest_path_moving):
                        closest_path_moving = item
            elif not moving and near_path:
                labels[2] = 1.0
                if self._item_is_closer(item, closest_path_stopped):
                    closest_path_stopped = item

        dynamic_side_actor = closest_side_moving is not None
        dynamic_turn_path_actor = False
        if closest_path_moving is not None and future_path_turns:
            path_distance, dist, _, _, lateral, _ = closest_path_moving
            dynamic_turn_path_actor = (
                path_distance < 2.5
                and dist < 25.0
                and abs(lateral) > 1.0
            )
        dynamic_conflict_actor = dynamic_side_actor or dynamic_turn_path_actor

        verified_lead_vehicle = closest_lead_vehicle
        lead_identifier = None
        if speed_reduced_by_obj_id not in (None, -1):
            lead_identifier = speed_reduced_by_obj_id
        elif vehicle_affecting_id not in (None, -1):
            lead_identifier = vehicle_affecting_id
        if lead_identifier is not None:
            lead_id_matches = (
                closest_lead_vehicle is not None
                and self._ids_match(closest_lead_vehicle[5].get('id'), lead_identifier)
            )
            verified_lead_vehicle = closest_lead_vehicle if lead_id_matches else None

        verified_lead_following = verified_lead_vehicle is not None
        verified_black_lead_following = (
            verified_lead_vehicle is not None
            and str(verified_lead_vehicle[5].get('color_name', '')).lower() == 'black'
        )

        verified_rule_or_static_stop = verified_traffic_stop or verified_stop_sign or verified_construction
        following_only = verified_lead_following and not dynamic_conflict_actor
        if dynamic_conflict_actor and expert_yields and not verified_rule_or_static_stop and not following_only:
            labels[3] = 1.0

        return {
            'actor_motion_labels': labels,
            'closest_path_moving': closest_path_moving,
            'closest_side_moving': closest_side_moving,
            'closest_path_stopped': closest_path_stopped,
            'closest_lead_vehicle': closest_lead_vehicle,
            'moving_actor_near_path': labels[0] > 0.0 or labels[1] > 0.0,
            'moving_junction_actor': moving_junction_actor,
            'dynamic_conflict_actor': dynamic_conflict_actor,
            'expert_yields': expert_yields,
            'future_path_turns': future_path_turns,
            'ego_near_junction': ego_near_junction,
            'verified_traffic_stop': verified_traffic_stop,
            'verified_green_light': verified_green_light,
            'verified_stop_sign': verified_stop_sign,
            'verified_construction': verified_construction,
            'verified_lead_following': verified_lead_following,
            'verified_black_lead_following': verified_black_lead_following,
        }

    @staticmethod
    def _motion_context_from_scene_facts(scene_facts):
        labels = scene_facts['actor_motion_labels']
        if labels[3] > 0.0:
            return 'A moving actor is close to the ego path, so the ego should yield until the path is clear.'
        closest_side_moving = scene_facts.get('closest_side_moving')
        if closest_side_moving is not None:
            _, _, obj_type, _, lateral, _ = closest_side_moving
            side_name = 'left' if lateral < 0 else 'right'
            return f'A moving {obj_type} from the {side_name} is close to the ego path.'
        closest_path_moving = scene_facts.get('closest_path_moving')
        if closest_path_moving is not None:
            _, _, obj_type, _, _, _ = closest_path_moving
            return f'A moving {obj_type} is close to the ego path.'
        closest_path_stopped = scene_facts.get('closest_path_stopped')
        if closest_path_stopped is not None:
            _, _, obj_type, _, _, _ = closest_path_stopped
            return f'A stopped {obj_type} is close to the ego path.'
        return ''

    def _get_actor_motion_labels(self, box_path, waypoints=None, commentary=None, current_measurement=None, scene_facts=None):
        """Build compact interaction labels from scene facts.

        The `commentary` argument is kept for older audit callers, but labels no
        longer depend on commentary text.
        """
        del commentary
        if scene_facts is None:
            boxes = self._load_boxes(box_path)
            if boxes is None:
                return None, ''
            scene_facts = self._build_interaction_scene_facts(boxes, waypoints, current_measurement)
        labels = scene_facts['actor_motion_labels'].copy()
        return labels, self._motion_context_from_scene_facts(scene_facts)

    @staticmethod
    def _sentence_claims_following(sentence_lower):
        following_phrases = (
            'stay behind',
            'remain behind',
            'hold your position behind',
            'hold your spot behind',
            'hold steady behind',
            'stay in place behind',
            'stay positioned behind',
            'maintain a spot behind',
            'keep a position behind',
            'do not advance past',
            'do not move past',
        )
        return (
            any(phrase in sentence_lower for phrase in following_phrases)
            or ('follow the' in sentence_lower and ('front' in sentence_lower or 'meter' in sentence_lower))
            or ('behind the' in sentence_lower and ('front' in sentence_lower or 'meter' in sentence_lower))
        )

    @staticmethod
    def _sentence_says_to_enter_or_speed_up(sentence_lower):
        speed_up_phrases = (
            'accelerate',
            'proceed',
            'drive through',
            'go through',
            'speed up',
            'drive faster',
            'gain speed',
            'advance faster',
            'boost your speed',
            'pick up speed',
            'pick up the pace',
            'increase velocity',
        )
        if any(phrase in sentence_lower for phrase in speed_up_phrases):
            return True
        if 'increase your speed' in sentence_lower:
            negated = (
                'do not increase your speed' in sentence_lower
                or "don't increase your speed" in sentence_lower
                or 'not increase your speed' in sentence_lower
            )
            return not negated
        return False

    def _remove_conflicting_junction_claims(self, commentary, actor_motion_labels=None, scene_facts=None):
        """Remove commentary sentences contradicted by verified scene facts."""
        if not commentary:
            return commentary

        moving_actor_near_path = False
        dynamic_conflict_actor = False
        expert_yields = False
        ego_interaction_context = False
        verified_green_light = False
        verified_traffic_stop = False
        verified_lead_following = False
        verified_black_lead_following = False
        moving_junction_actor = False
        strict_yield_interaction = False

        if scene_facts is not None:
            moving_actor_near_path = bool(scene_facts.get('moving_actor_near_path'))
            dynamic_conflict_actor = bool(scene_facts.get('dynamic_conflict_actor'))
            expert_yields = bool(scene_facts.get('expert_yields'))
            ego_interaction_context = bool(scene_facts.get('ego_near_junction') or scene_facts.get('future_path_turns'))
            verified_green_light = bool(scene_facts.get('verified_green_light'))
            verified_traffic_stop = bool(scene_facts.get('verified_traffic_stop'))
            verified_lead_following = bool(scene_facts.get('verified_lead_following'))
            verified_black_lead_following = bool(scene_facts.get('verified_black_lead_following'))
            moving_junction_actor = bool(scene_facts.get('moving_junction_actor'))
            labels = scene_facts.get('actor_motion_labels')
            if labels is not None and len(labels) > 3:
                strict_yield_interaction = labels[3] > 0.0
        elif actor_motion_labels is not None:
            moving_actor_near_path = actor_motion_labels[0] > 0.0 or actor_motion_labels[1] > 0.0
            dynamic_conflict_actor = actor_motion_labels[3] > 0.0
            strict_yield_interaction = actor_motion_labels[3] > 0.0

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
        sentences = re.split(r'(?<=[.!?])\s+', commentary.strip())
        kept = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            says_to_enter = self._sentence_says_to_enter_or_speed_up(sentence_lower)
            has_stale_claim = any(claim in sentence_lower for claim in stale_junction_claims)
            mentions_green = 'traffic light is green' in sentence_lower
            mentions_red = 'red traffic light' in sentence_lower or 'traffic light is red' in sentence_lower
            following_claim = self._sentence_claims_following(sentence_lower)
            black_car_following_claim = 'black car' in sentence_lower and any(
                phrase in sentence_lower
                for phrase in ('front', 'behind', 'stay behind', 'follow the')
            )

            stale_junction_conflict = has_stale_claim and (
                moving_actor_near_path
                or moving_junction_actor
                or dynamic_conflict_actor
                or (expert_yields and ego_interaction_context)
            )
            green_light_conflict = mentions_green and says_to_enter and (
                moving_actor_near_path
                or moving_junction_actor
                or dynamic_conflict_actor
                or (expert_yields and ego_interaction_context)
            )
            strict_green_light_conflict = mentions_green and strict_yield_interaction
            false_red_light = mentions_red and verified_green_light
            false_green_light = mentions_green and verified_traffic_stop
            unsupported_following = following_claim and (
                strict_yield_interaction or not verified_lead_following
            )
            unsupported_black_following = black_car_following_claim and (
                strict_yield_interaction or not verified_black_lead_following
            )
            explicit_go_conflict = says_to_enter and dynamic_conflict_actor and expert_yields

            if (
                stale_junction_conflict
                or green_light_conflict
                or strict_green_light_conflict
                or false_red_light
                or false_green_light
                or unsupported_following
                or unsupported_black_following
                or explicit_go_conflict
            ):
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
        interaction_scene_facts = None
        if getattr(self, 'use_motion_descriptions', False):
            boxes = self._load_boxes(current_box_path)
            if boxes is not None:
                interaction_scene_facts = self._build_interaction_scene_facts(
                    boxes,
                    data['waypoints_org'],
                    current_measurement,
                )
                actor_motion_labels, actor_motion_context = self._get_actor_motion_labels(
                    current_box_path,
                    data['waypoints_org'],
                    current_measurement=current_measurement,
                    scene_facts=interaction_scene_facts,
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
                    if getattr(self, 'use_motion_descriptions', False) and interaction_scene_facts is not None:
                        commentary = self._remove_conflicting_junction_claims(
                            commentary,
                            actor_motion_labels,
                            interaction_scene_facts,
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
