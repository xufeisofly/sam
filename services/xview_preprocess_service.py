# coding: utf-8
import os
import json
import numpy as np
from services.base_preprocess_service import BasePreprocessService
from schemas.preprocess_result import PreprocessResult, PreprocessResultItem, BoxItem
from util.constant import ORIGIN_DATA_DIR, DataType


class xViewPreprocessService(BasePreprocessService):
    def __init__(self, dataset_path=None) -> None:
        self._dataset_path = os.path.join(ORIGIN_DATA_DIR, "xView") if dataset_path is None else dataset_path
    
    def _get_data_paths(self, data_type: str):
        image_dir = {
            "train": "train_images"
        }[data_type]

        geojson_dir = {
            "train": "train_annotations"
        }[data_type]

        image_path = os.path.join(self._dataset_path, image_dir)
        geojson_path = os.path.join(self._dataset_path, geojson_dir)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image directory {image_path} does not exist.")
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"Annotation directory {geojson_path} does not exist.")

        return image_path, geojson_path


    def process_data_type(self, data_type: str, limit=-1) -> PreprocessResult:
        image_path, geojson_path = self._get_data_paths(data_type)

        all_geojson_files = os.listdir(geojson_path)
        all_geojson_files = [f for f in all_geojson_files if f.endswith('.geojson')]

        result = PreprocessResult()

        for geojson_file in all_geojson_files[:limit] if limit > 0 else all_geojson_files:
            with open(os.path.join(geojson_path, geojson_file), 'r') as file:
                geojson_data = json.load(file)

            img_file_name = os.path.splitext(geojson_file)[0] + ".tif"
            img_file_path = os.path.join(image_path, img_file_name)
            if not os.path.exists(img_file_path):
                print(f"Warning: Image file {img_file_path} not found.")
                continue

            result_item = PreprocessResultItem(img_file_path=img_file_path, data_type=DataType[data_type.upper()])

            for feature in geojson_data['features']:
                pixel_coords = feature['properties']['bounds_imcoords']
                box_array = np.array([int(num) for num in pixel_coords.split(",")])
                box_item = BoxItem(
                    ori_label=self._get_ori_label_by_token(feature['properties']['type_id']),
                    box_array=np.array(box_array)
                )
                result_item.append(box_item)    

            result.append(result_item)
        
        return result


    def call(self, limit=-1) -> dict:
        train_result = self.process_data_type("train",limit)
        
        result = PreprocessResult(
            train_result_list=train_result.result_list)
        return result
        


    def ori_label_2_id_map(self) -> dict:
        return {
            "Fixed-wing aircraft" : 255,
            "Small aircraft" : 254,
            "Passenger or Cargo Plane" : 253,
            "Helicopter" : 252,
            "Passenger Vehicle" : 251,
            "Small car" : 250,
            "Bus" : 249,
            "Pickup Truck" : 248,
            "Utility Truck" : 247,
            "Truck" : 246,
            "Cargo Truck" : 245,
            "Truck Tractor with Box Trailer" : 244,
            "Truck Tractor" : 243,
            "Trailer" : 242,
            "Truck Tractor with Flatbed Trailer" : 241,
            "Truck Tractor with Liquid Tank" : 240,
            "Crane Truck" : 239,
            "Railway Vehicle" : 238,
            "Passenger Car" : 237,
            "Cargo or Container Car" : 236,
            "Flat Car" : 235,
            "Tank Car" : 234,
            "Locomotive" : 233,
            "Maritime Vessel" : 232,
            "Motorboat" : 231,
            "Sailboat" : 230,
            "Tugboat" : 229,
            "Barge" : 228,
            "Fishing Vessel" : 227,
            "Ferry" : 226,
            "Yacht" : 225,
            "Container Ship" : 224,
            "Oil Tanker" : 223,
            "Engineering Vehicle" : 222,
            "Tower Crane" : 221,
            "Container Crane" : 220,
            "Reach Stacker" : 219,
            "Straddle Carrier" : 218,
            "Mobile Crane" : 217,
            "Dump Truck" : 216,
            "Haul Truck" : 215,
            "Tractor" : 214,
            "Front Loader or Bulldozer" : 213,
            "Excavator" : 212,
            "Cement Mixer" : 211,
            "Ground Grader" : 210,
            "Hut or Tent" : 209,
            "Shed" : 208,
            "Building" : 207,
            "Aircraft Hangar" : 206,
            "75" : 205,
            "Damaged or Demolished Building" : 204,
            "Facility" : 203,
            "Construction Site" : 202,
            "82" : 201,
            "Vehicle Lot" : 200,
            "Helipad" : 199,
            "Storage Tank" : 198,
            "Shipping Container Lot" : 197,
            "Shipping Container" : 196,
            "Pylon" : 195,
            "Tower" : 194
        }

    def _get_ori_label_by_token(self, label_token: int) -> str:
        token2ori = {
            11 : "Fixed-wing aircraft",
            12 : "Small aircraft",
            13 : "Passenger or Cargo Plane",
            15 : "Helicopter",
            17 : "Passenger Vehicle",
            18 : "Small car",
            19 : "Bus",
            20 : "Pickup Truck",
            21 : "Utility Truck",
            23 : "Truck",
            24 : "Cargo Truck",
            25 : "Truck Tractor with Box Trailer",
            26 : "Truck Tractor",
            27 : "Trailer",
            28 : "Truck Tractor with Flatbed Trailer",
            29 : "Truck Tractor with Liquid Tank",
            32 : "Crane Truck",
            33 : "Railway Vehicle",
            34 : "Passenger Car",
            35 : "Cargo or Container Car",
            36 : "Flat Car",
            37 : "Tank Car",
            38 : "Locomotive",
            40 : "Maritime Vessel",
            41 : "Motorboat",
            42 : "Sailboat",
            44 : "Tugboat",
            45 : "Barge",
            47 : "Fishing Vessel",
            49 : "Ferry",
            50 : "Yacht",
            51 : "Container Ship",
            52 : "Oil Tanker",
            53 : "Engineering Vehicle",
            54 : "Tower Crane",
            55 : "Container Crane",
            56 : "Reach Stacker",
            57 : "Straddle Carrier",
            59 : "Mobile Crane",
            60 : "Dump Truck",
            61 : "Haul Truck",
            62 : "Tractor",
            63 : "Front Loader or Bulldozer",
            64 : "Excavator",
            65 : "Cement Mixer",
            66 : "Ground Grader",
            71 : "Hut or Tent",
            72 : "Shed",
            73 : "Building",
            74 : "Aircraft Hangar",
            75 : "75",
            76 : "Damaged or Demolished Building",
            77 : "Facility",
            79 : "Construction Site",
            82 : "82",
            83 : "Vehicle Lot",
            84 : "Helipad",
            86 : "Storage Tank",
            89 : "Shipping Container Lot",
            91 : "Shipping Container",
            93 : "Pylon",
            94 : "Tower"
        }
        return token2ori[label_token]